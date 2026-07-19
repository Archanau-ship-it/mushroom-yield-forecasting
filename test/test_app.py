from app import predict_yield


def test_predict_returns_float():

    result = predict_yield(
        22,
        88,
        900,
        500
    )

    assert isinstance(result, float)


def test_prediction_range():

    result = predict_yield(
        22,
        88,
        900,
        500
    )

    assert 0 < result < 50