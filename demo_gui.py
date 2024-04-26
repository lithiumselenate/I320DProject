import gradio as gr
import demo 
import training_preparation

inputs=gr.Image(label="Input Image Component", type="pil")
output = gr.Textbox(label="Prediction")

gr.Interface(
    fn=demo.get_single,
    inputs=inputs,
    outputs=output,
    title="Handwriting recognition",
    description="Select File and click on submit to get the prediction."
).launch(debug = True)