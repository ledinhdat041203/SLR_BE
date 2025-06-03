import torch
from LSTM_ATTENTION_RESIDUAL import LSTMModel, MultiHeadAttention

NUM_CLASS = 63
TOTAL_POSE_LANDMARKS = 25
TOTAL_HAND_LANDMARKS = 21
TOTAL_HANDS = 2
NUM_FRAME_PROCESS = 32
TOTAL_COORDINATES = TOTAL_POSE_LANDMARKS * 3 + TOTAL_HAND_LANDMARKS * 3 * TOTAL_HANDS
fixed_size = (256, 256)
NOSE_POSITION=0

model = LSTMModel(input_size=67*3, hidden_size=128, num_layers=3, num_classes=NUM_CLASS)
state_dict = torch.load("best_model_residual_lstm_attention.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 3. Convert sang TorchScript (.pt)
# example_input = torch.randn(1, 30, 21, 3)  
example_input = torch.randn(1, NUM_FRAME_PROCESS, 67, 3)
traced_script_module = torch.jit.trace(model, example_input)  

# 4. Lưu lại file .pt
traced_script_module.save("best_model_residual_lstm_attention.pt")
print("Saved model.pt successfully.")
