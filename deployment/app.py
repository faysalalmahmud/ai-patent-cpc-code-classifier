#import necessery libraries
import gradio as gr
import onnxruntime as rt
from transformers import AutoTokenizer
import torch, json

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

with open("encode_revised_cpc_codes.json", "r") as fp:
  encode_cpc_types = json.load(fp)

cpc_code = list(encode_cpc_types.keys())

inf_session = rt.InferenceSession('distilroberta-base-patent-cpc-classifier-quantized.onnx')
input_name = inf_session.get_inputs()[0].name
output_name = inf_session.get_outputs()[0].name

def classify_cpc_code(abstract):
  input_ids = tokenizer(abstract)['input_ids'][:512]
  logits = inf_session.run([output_name], {input_name: [input_ids]})[0]
  logits = torch.FloatTensor(logits)
  probs = torch.sigmoid(logits)[0]
  return dict(zip(cpc_code, map(float, probs))) 

label = gr.Label(num_top_classes=5)
iface = gr.Interface(
    fn=classify_cpc_code,
    inputs="text",
    outputs=label, 
    examples=
      ["An artificial intelligence moving agent is launched. The artificial intelligence moving agent according to an embodiment of the present invention includes a camera that captures an image, and an object, and provides an image of the object to an artificial intelligence model to obtain information on the type of the object, and the object and a processor for acquiring correction type information designated by a user with respect to the captured image, and training the artificial intelligence model by using the correction type information."]
    )
iface.launch(inline=False)