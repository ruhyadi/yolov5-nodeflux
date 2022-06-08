# Simplify dockerfile

FROM ultralytics/yolov5:latest-cpu

RUN mkdir /opt/nuclio

COPY ./weights /opt/nuclio/weights