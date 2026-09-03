def build_result(
        algorithm,
        defect_type,
        result,
        metrics=None
):
    if metrics is None:
        metrics = {}

    return {

        "algorithm": algorithm,

        "defect_type": defect_type,

        "status": "success",

        "result": result,

        "metrics": metrics

    }
