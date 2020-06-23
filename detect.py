import argparse
from flask import Flask, make_response, request, jsonify, current_app

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)


def detect(source):
    imgsz = (
        (320, 192) if ONNX_EXPORT else opt.img_size
    )  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    weights = opt.weights

    # Initialize
    device = torch_utils.select_device(device="cpu" if ONNX_EXPORT else opt.device)

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith(".pt"):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)["model"])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split(".")[-1], "onnx")  # *.onnx filename
        torch.onnx.export(
            model,
            img,
            f,
            verbose=False,
            opset_version=11,
            input_names=["images"],
            output_names=["classes", "boxes"],
        )

        # Validate exported model
        import onnx

        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(
            onnx.helper.printable_graph(model.graph)
        )  # Print a human readable representation of the graph
        return

    # Set Dataloader

    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.float()) if device.type != "cpu" else None  # run once
    path, img, im0s = dataset.readImage()
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    t2 = torch_utils.time_synchronized()

    # Apply NMS
    pred = non_max_suppression(
        pred,
        opt.conf_thres,
        opt.iou_thres,
        multi_label=False,
        classes=opt.classes,
        agnostic=opt.agnostic_nms,
    )

    # Apply Classifier

    # Process detections
    for i, det in enumerate(pred):  # detections for image i

        p, s, im0 = path, "", im0s

        s += "%gx%g " % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += "%g %ss, " % (n, names[int(c)])  # add to string

            # Write results
            coordinate = ""

            for *xyxy, conf, cls in det:
                if int(cls) == 154:
                    return (
                        str(xyxy)
                        .replace("tensor(", "")
                        .replace("[", "")
                        .replace(".)", "")
                        .replace("]", "")
                    )

        # Print time (inference + NMS)
        print("%sDone. (%.3fs)" % (s, t2 - t1))
    return "not found"
    print("Done. (%.3fs)" % (time.time() - t0))


@app.route("/", methods=["GET", "POST"])
def index():
    return "<h1> Deployed to Heroku</h1>"


@app.route("/getimage", methods=["POST"])
def getimage():
    try:
        data = (request.data.decode()).split(",")[1]
        body = base64.decodebytes(data.encode("utf-8"))
        img = Image.open(BytesIO(body))
        img = np.array(img)

        BGR_IMG = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        print(BGR_IMG.shape)

        with torch.no_grad():
            return detect(BGR_IMG)

    except Exception as e:
        print(e)
        return "error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    opt.cfg = "500.cfg"
    opt.iou_thres = 0.6
    opt.names = "500.names"
    opt.conf_thres = 0.3
    opt.device = ""
    opt.img_size = 512
    opt.agnostic_nms = False
    opt.augment = False
    opt.classes = None
    opt.weights = "500.pt"
    app.run()

