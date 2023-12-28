FROM alpine
RUN apk add python3 py3-numpy py3-scikit-learn

COPY . /apps
WORKDIR /apps
CMD ["python3", "model_iris.py"]

