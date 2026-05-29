# code/tools/export_onnx.py
import os, argparse, torch, json
from code.models.inception_time import InceptionTime

def main(ckpt, out):
    os.makedirs(out, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu")
    classes = ck["classes"]; c_out = len(classes)
    m = InceptionTime(c_in=12, c_out=c_out)
    m.load_state_dict(ck["model"])
    m.eval()
    x = torch.zeros(1,12,5000, dtype=torch.float32)
    onnx_path = os.path.join(out, "inception_5cls.onnx")
    torch.onnx.export(
        m, x, onnx_path,
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0:"B", 2:"L"}, "logits": {0:"B"}},
        opset_version=17
    )
    json.dump(classes, open(os.path.join(out,"classes_5cls.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("EXPORTED", onnx_path, "classes", classes)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="O:/ECG AI project/models/inception_5cls_best.pt")
    ap.add_argument("--out",  default="O:/ECG AI project/models")
    a=ap.parse_args(); main(a.ckpt, a.out)
