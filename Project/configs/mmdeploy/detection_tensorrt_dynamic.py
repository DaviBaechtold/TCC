
_base_ = ['mmdeploy://base/base_static.py']

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['boxes', 'labels'],
    input_shape=(1, 3, 640, 640),
    dynamic_axes={
        'input': {0: 'batch'},
        'boxes': {0: 'batch'},
        'labels': {0: 'batch'}
    }
)

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True,
        max_workspace_size=1 << 30
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=(1, 3, 640, 640),
                    opt_shape=(1, 3, 640, 640),
                    max_shape=(8, input_shape[1], input_shape[2], input_shape[3])
                )
            )
        )
    ]
)

codebase_config = dict(
    type='mmdet',
    task='ObjectDetection'
)
