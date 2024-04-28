import gradio as gr
import demo 
import training_preparation

inputs=gr.Image(label="Input Image", type="pil")
output1 = gr.Textbox(label="Prediction")

gr.Interface(
    fn=demo.get_single,
    inputs=inputs,
    outputs=output1,
    title="Handwriting recognition",
    description="Select File and click on submit to get the prediction."
).launch(share=True)