## With NMS Plugin

###  Generate trt File [More Details](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/README.md)
```shell
python export.py -o mdoel.onnx -e model.trt --end2end --v8 only for yolov8
```

- [x] YOLOv8
- [x] YOLOv7
- [x] YOLOv6
- [x] YOLOX
- [x] YOLOv5
- [x] YOLOv3 


**Edit code for your model**

```c++
auto in_dims = engine->getBindingDimensions(engine->getBindingIndex("image_arrays"));
```
**run**

```shell
cd end2end && mkdir build && build
cmake ..
make
./yolo -model_path  engine   -image_path xxx.jpg #for images

./yolo -model_path  engine   -video_path xxx.mp4 #for videos
```
