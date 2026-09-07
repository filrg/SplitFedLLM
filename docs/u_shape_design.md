# Thiết kế U-shape cho SplitFedLLM

Trạng thái: runtime U-shape cơ bản đã triển khai trong `src/fine_tune/u_shape.py`,
`src/model/u_shape.py` và `src/UShapeServer.py`. README mô tả chính xác cấu hình và
hành vi hiện có. Tài liệu này vẫn giữ các hướng tối ưu mở rộng làm mục tiêu thiết kế.

Phạm vi bản triển khai: ba model hiện có với weights untied, route cố định theo
round, credit cố định, cửa sổ đồng bộ theo group, objective mean microbatch,
validation loss local trước FedAvg, và truyền tensor qua RabbitMQ. Chưa triển khai
adaptive load balancing, batching nhiều owner, tensor data plane riêng hoặc tự
khôi phục model/optimizer/RNG sau lỗi process. Lỗi sẽ abort round và yêu cầu
restart từ checkpoint của round đã hoàn tất.

## 1. Mục tiêu và ranh giới privacy

Token IDs, text, labels, logits, predictions và loss từng mẫu ở lại thiết bị sở hữu
dữ liệu (owner). Thiết bị tính toán từ xa (worker) chỉ nhận hidden states, gradient
của hidden states và metadata điều phối. Coordinator quản lý phiên, topology và
federated aggregation; không nhận dữ liệu huấn luyện thô.

U-shape loại bỏ việc truyền trực tiếp dữ liệu/label, nhưng không chứng minh rằng
activation, gradient hoặc model update không thể bị dùng để suy luận lại dữ liệu.
Độ dài chuỗi/padding cũng là metadata có thể lộ. Bảo vệ trước các cuộc tấn công suy
luận cần threat model và đánh giá riêng; TLS chỉ bảo vệ trên đường truyền.

Không chọn thiết kế worker tính logits rồi owner trả gradient logits: với
cross-entropy chuẩn, mỗi token hợp lệ có dL/dlogits = softmax(logits) - one_hot(y),
nhân hệ số chuẩn hóa dương. Worker có thể suy ra nhãn từ gradient này.

## 2. Phân chia model

Một đường tính toán gồm ba vai trò logic, dù chỉ cần hai thiết bị vật lý:

```mermaid
sequenceDiagram
    participant O as Owner: front + tail + labels
    participant W as Worker: transformer body
    O->>O: a = front(input_ids), giữ graph a
    O->>W: BODY_FORWARD(id, a.detach(), metadata)
    W->>W: a_leaf.requires_grad; h = body(a_leaf), giữ graph h
    W->>O: TAIL_FORWARD(id, h.detach())
    O->>O: h_leaf.requires_grad; logits = tail(h_leaf)
    O->>O: loss(logits, labels).backward()
    O->>W: BODY_BACKWARD(id, h_leaf.grad)
    W->>W: h.backward(grad_h), lấy a_leaf.grad
    W->>O: FRONT_BACKWARD(id, a_leaf.grad)
    O->>O: a.backward(grad_a), hoàn tất microbatch
```

| Model | Owner/front | Worker/body | Owner/tail |
|---|---|---|---|
| GPT-2 | Embedding, positional embedding, block đầu | Block còn lại | Final LayerNorm, LM head, loss |
| Llama | Embedding, block đầu | Block còn lại | RMSNorm cuối, LM head, loss |
| BERT | Embedding, encoder đầu | Encoder còn lại | Pooler, dropout, classifier, loss |

Có thể đưa thêm block cuối vào tail nếu tài nguyên owner cho phép. Với mô hình
tie embedding/head weights, giữ đúng một Parameter được chia sẻ trên owner và
chỉ đăng ký nó một lần trong optimizer. LoRA cũng chia theo nơi đặt module.
Tính loss/shift target giữ tương thích adapter hiện tại trong bước chuyển đổi;
sửa objective Llama là một thay đổi độc lập cần test riêng.

Head có thể tốn đáng kể bộ nhớ và compute vì trọng số cỡ hidden_size × vocab_size.
Do đó phải profile owner; không mặc định tail là một tác vụ rẻ.

## 3. State và giao thức

Mọi message chứa protocol_version, round_id, group_id, window_id, microbatch_id,
owner_id, worker_id, model_version, message_type, shape và dtype cần thiết.
ID duy nhất có thể dùng tuple (round_id, owner_id, window_id, microbatch_id).

| Loại message | Tuyến | Tensor |
|---|---|---|
| BODY_FORWARD | owner → worker đã chọn | a |
| TAIL_FORWARD | worker → đúng owner | h |
| BODY_BACKWARD | owner → đúng worker giữ graph | grad_h |
| FRONT_BACKWARD | worker → đúng owner giữ graph | grad_a |
| CREDIT / WINDOW_CLOSE / DRAINED / COMMIT | Control plane | Không có tensor dữ liệu |

Serialization dùng schema cho từng loại message, whitelist trường; bỏ hoàn toàn
label khỏi payload hiện tại. Không serialize nguyên batch. Causal mask tái tạo
ở worker từ shape và padding metadata tối thiểu, tránh gửi mask B×1×T×T. Với
packing hoặc attention đặc biệt phải truyền đủ segment/position metadata để giữ
đúng ngữ nghĩa; không được thay bằng causal mask đơn giản một cách mặc định.

Owner state: labels local, graph a, trạng thái tail/backward, route và version.
Worker state: a_leaf, graph h, route và version. Giữ graph đến khi backward tương
ứng hoàn tất, detach duy nhất tại ranh giới truyền tensor. Không forward lại ở
front hay body trong luồng mặc định. Xóa state/tensor ngay khi không còn cần.

Gradient phải về đúng worker đã thực hiện forward. Không dùng một gradient queue
chung có competing consumers cho mọi replica. Các queue inbox định danh node và
message_type; credit tính theo graph đang sống, không theo số message đã ACK.

## 4. Scheduler event-driven và parallelism

Một scheduler chung có các handler on_body_forward, on_tail_forward,
on_body_backward, on_front_backward. Model adapter chỉ thực hiện front/body/tail,
chuẩn bị metadata và loss. Scheduler không chứa nhánh riêng theo tên LLM.

Tách hai tham số:

- max_inflight = C: giới hạn graph đang sống, dùng backpressure theo bộ nhớ.
- microbatches_per_window = K: số microbatch tích lũy trước optimizer.step.

K có thể lớn hơn C. Trong cùng một window, mỗi khi một microbatch hoàn tất và trả
credit thì owner có thể phát microbatch mới, đến quota K. Đây là khác biệt so với
vòng hiện tại vốn phát đủ control-count rồi chờ drain toàn bộ mới phát tiếp.

Chính sách đề xuất là lịch dựa trên readiness, theo hướng 1F1B:

1. Worker ưu tiên backward đã sẵn sàng để giải phóng graph, xen kẽ forward hợp lệ.
2. Owner xử lý front backward và tail forward/backward đang chờ trước khi nạp
   thêm front forward nếu bộ nhớ áp lực. Tail phải được phục vụ thường xuyên để
   worker nhận grad_h và tiếp tục pipeline.
3. Dùng aging/quota giữa owner và loại tác vụ để tránh starvation. Không ép
   luân phiên 1F1B khi một dependency chưa sẵn sàng.
4. Chỉ nhận forward khi có credit ở cả owner và worker; giải phóng credit sau khi
   hoàn tất lifetime graph tương ứng. Dành sẵn bộ nhớ cho tác vụ backward/tail.
5. Sau warmup, worker có thể forward B trong lúc owner tính tail A; sau đó worker
   backward A trong lúc owner forward C. Nhiều owner cung cấp thêm việc khi một
   owner đang đợi mạng hoặc compute.

Mỗi GPU mặc định có một compute executor để tránh race khi ghi Parameter.grad.
I/O, serialize và copy có thể chạy bất đồng bộ, dùng bounded queues, pinned memory
và CUDA events khi phù hợp. Không chia sẻ Pika BlockingConnection/channel tùy ý
giữa các thread. ACK/credit và quyền sở hữu tensor buffer phải được xử lý trước
khi buffer được tái sử dụng. Pipeline overlap giữa thiết bị không đòi hỏi chạy
hai backward đồng thời trên cùng một GPU.

Có thể batch các BODY_FORWARD sẵn sàng cùng model_version và length bucket từ
nhiều owner. Phải giữ mapping row → microbatch và trả kết quả đúng owner. Batching
body tạo graph chung: cần giữ graph và chờ đủ gradient thành viên để backward
một lần, hoặc thiết kế autograd tách biệt; không giải phóng graph sau gradient
đầu tiên. Đây là tối ưu giai đoạn sau vì có thể gây head-of-line blocking.

## 5. Optimizer và nhiều worker

Không optimizer.step trong khi còn graph dùng phiên bản trọng số hiện tại.
Quy tắc này áp dụng cho cả front, tail và body. Owner dùng một optimizer cho
front+tail (hoặc hai optimizer với cùng commit barrier); không step tail ngay
sau loss khi front còn chờ grad_a.

Một worker replica và tập owner được gán cho nó tạo thành một training group.
Worker dùng chung trọng số body cho các owner trong group. Vì vậy barrier phải
xét toàn bộ graph của group, không chỉ graph của một owner.

Quy trình mỗi window:

1. Coordinator group cấp version v và quota microbatch hữu hạn cho từng owner.
2. Tất cả module giữ trọng số ở version v, accumulate gradient trong window.
3. Khi nhận đủ quota (hoặc owner báo hết dữ liệu), đóng nhận forward mới cho v.
4. Drain tất cả backward, nhận DRAINED từ các owner và worker, kiểm tra pending=0.
5. Chuẩn hóa gradient và COMMIT đúng một lần; chuyển sang version v+1.

Định nghĩa objective và hệ số chuẩn hóa trước khi chạy. Với objective mean token,
dùng tổng loss trên token hợp lệ chia tổng token hợp lệ của window/group. Cùng
một hệ số phải áp dụng cho gradient của cả ba phần; không chia K lặp lại tại từng
stage. Với objective trung bình theo client, dùng trọng số client rõ ràng. Tổng
token/count là metadata tùy chọn cần xét trong threat model, không gửi label.

Các group độc lập có thể chạy song song và FedAvg ở cuối round, giữ tinh thần
testbed hiện tại. Không coi các replica body độc lập là cùng một model đồng bộ
từng step. Muốn objective đồng bộ toàn cục thì cần gradient synchronization và
barrier liên group, chấp nhận chi phí đồng bộ bổ sung.

Gán owner → group theo round và cân bằng lại ở ranh giới checkpoint/round. Mỗi
microbatch luôn pin route suốt lifetime. Không chuyển công việc đang có graph
sang replica khác. Replica nhanh nhận nhiều owner/quota hơn dựa trên số đo
throughput; không round-robin bỏ qua tải và version.

## 6. Hoàn tất round, lỗi và aggregation

NOTIFY chỉ gửi khi owner đã xong toàn bộ FRONT_BACKWARD, state rỗng và window đã
commit. Coordinator xác nhận cả worker đã drain trước khi yêu cầu UPDATE.

Checkpoint và aggregation theo key gốc cùng role front/body/tail. Owner gửi hai
phần local; worker gửi body. Không nhân đôi trọng số aggregation do owner giữ
hai role. Weight tính theo objective đã chọn và lượng dữ liệu thực sự xử lý,
không theo số message. Mapping khôi phục full model cần test độc lập. Validation
trên dữ liệu private chạy ở owner; không chuyển labels lên coordinator để eval.

Dùng manual ACK, state transition có kiểm tra ID/version, duplicate suppression
và COMMIT idempotent. ACK riêng lẻ không đảm bảo exactly-once cho optimizer.
Nếu process chết thì graph mất: hủy window bị ảnh hưởng, phục hồi checkpoint
đồng nhất gồm model/optimizer/RNG và replay window ở owner. Không retry backward
vào graph đã bị giải phóng và không công bố window thành công khi thiếu client.

## 7. Chi phí và chọn cấu hình theo số đo

Đặt S=B×T×H×bytes_per_element, giả sử hai cut có cùng H và dtype. U-shape truyền
xấp xỉ 4S mỗi microbatch: a, h, grad_h, grad_a, chưa tính metadata. Hai-stage cũ
truyền khoảng 2S cộng labels. U-shape thêm một lượt trao đổi; không mặc định nhanh
hơn kiến trúc cũ. So với U-shape trả logits kích thước B×T×V, trả hidden states
giảm payload nhánh cuối theo tỷ lệ V/H. Ví dụ V=32000, H=768 thì khoảng 41.7 lần
cho riêng nhánh đó, không phải tăng tốc end-to-end 41.7 lần.

Ước lượng ban đầu C ≈ ceil(latency vòng microbatch / khoảng cách phát microbatch
mục tiêu), sau đó chặn bởi ngân sách activation memory ở cả hai phía. Sweep C,
K, microbatch size, cut placement và số replica bằng benchmark thực tế. Chọn
cấu hình có tokens/s tốt và p95 latency/peak memory trong giới hạn; không có một
giá trị control-count tối ưu cho mọi thiết bị.

Giữ RabbitMQ để kiểm chứng giao thức ban đầu. Nếu profile chỉ ra CPU serialization
hoặc broker là bottleneck, tách control plane RabbitMQ khỏi data plane tensor
trực tiếp (chọn transport theo mạng/hardware thực tế). Tối ưu dtype cần kiểm tra
sai số/hội tụ, không tự động nén mất mát gradient.

## 8. Các thay đổi triển khai và tiêu chí nghiệm thu

- model/{Bert,GPT2,Llama}.py: tách front/body/tail, giữ state_dict mapping, LoRA,
  tied weights và API inference phù hợp.
- fine_tune/adapters.py: contract ba vai trò, local labels/loss; không truyền label.
- fine_tune/scheduler.py: event loop, bốn tensor message, graph cache hai phía,
  credits, group window/version và commit barrier.
- RpcClient.py: owner chạy front+tail, worker chạy body; config hai cut khi cần.
- Server.py và Utils.py: topology/role, group membership, hoàn tất round và ghép
  front/body/tail khi FedAvg/checkpoint; validation private tại owner.

Thứ tự triển khai: một owner/một worker/C=1 → pipeline C>1 → nhiều owner dùng chung
worker → nhiều group/FedAvg → batching/transport optimization sau profiling.

Kiểm thử bắt buộc:

1. So sánh loss, gradient, update với monolithic model cùng seed/dropout/objective.
2. Mỗi phần chỉ forward một lần/microbatch, graph được giải phóng sau backward.
3. Capture mọi payload: không có input token IDs, label, logits hay prediction.
4. Gradient đảo thứ tự, duplicate, stale version, timeout, empty/partial window.
5. Nhiều owner chung worker: không step khi bất kỳ graph nào còn pending.
6. C giới hạn memory/inflight, K>C thực sự có overlap và không deadlock/starvation.
7. FedAvg, state_dict reconstruction, LoRA merge và tied weights không nhân đôi.
8. Benchmark RabbitMQ thật trên thiết bị mục tiêu: tokens/s, p50/p95 latency,
   GPU utilization, peak memory, wire bytes, queue wait, bubble và commit time.

## Nguồn tham khảo

- Label leakage qua gradient: https://arxiv.org/abs/2102.08504
- PyTorch pipeline schedules và microbatching:
  https://docs.pytorch.org/docs/stable/distributed.pipelining.html
- PyTorch autograd, graph lifetime và concurrency:
  https://docs.pytorch.org/docs/main/notes/autograd.html

Thiết kế group, protocol và placement phía trên là đề xuất cho repository này;
chưa phải kết quả benchmark hay cam kết bảo mật từ các nguồn tham khảo.
