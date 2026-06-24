from src.inference import predict_image

def test_prediction():
    result = predict_image("data/test/fire/sample.jpg")
    assert result in ["fire", "smoke", "normal"]
