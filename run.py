import dvc.api
import fire
import mlflow
import mlflow.pyfunc
import numpy as np
import onnx
import onnxruntime
import pandas as pd
from sklearn.model_selection import train_test_split


class ONNXInference:
    def __init__(self):
        self.onnx_model_path = "model/rf_iris.onnx"
        self.model = None

    def load_model(self):
        # Load the ONNX model
        self.model = onnx.load(self.onnx_model_path)

    def predict(self):
        # Initialize ONNX Runtime Inference Session
        sess = onnxruntime.InferenceSession(self.onnx_model_path)

        # Prepare the input data
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        with dvc.api.open("data/iris_dataset.csv") as fd:
            df = pd.read_csv(fd)
        X = df.drop("target", axis=1)
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2
        )

        # Run the inference
        output = sess.run(
            [label_name], {input_name: np.array(X_test).astype(np.double)}
        )[0]

        return output


def run_server():
    onnx_inference = ONNXInference()
    onnx_inference.load_model()

    # Run inference on the provided input data
    # result = onnx_inference.predict()

    with mlflow.start_run():
        onnx_model_path = "model/rf_iris.onnx"
        mlflow.log_params({"onnx_model_path": onnx_model_path})
        mlflow.log_artifact(onnx_model_path)


if __name__ == "__main__":
    fire.Fire(run_server)
