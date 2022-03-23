import torch
from .observation_type import ObservationType
import torch.nn.qat as nnqat
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.qat as nniqat

from ...fuser_method_mappings import reverse_sequential_wrapper2

def get_native_backend_config_dict():
    """ Get backend for PyTorch Native backend_config_dict (fbgemm/qnnpack)
    """
    # dtype configs

    # weighted op int8 config
    # activation: quint8, weight: qint8, bias: float
    weighted_op_int8_dtype_config = {
        # optional, input activation dtype
        "input_dtype": torch.quint8,
        # optional, weight dtype
        "weight_dtype": torch.qint8,
        # optional, bias dtype
        "bias_dtype": torch.float,
        # optional, output activation dtype
        "output_dtype": torch.quint8
    }
    # operator (module/functional/torch ops) configs
    linear_module_config = {
        # Please see README under this folder for pattern format
        "pattern": torch.nn.Linear,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        # the root module for the pattern, used to query the reference quantized module
        # e.g. for a (torch.nn.ReLU, torch.nn.Linear) pattern, the root will be torch.nn.Linear
        "root_module": torch.nn.Linear,
        # the corresponding reference quantized module for the root module
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
        "qat_module": nnqat.Linear,
    }
    linear_qat_config = {
        "pattern": nnqat.Linear,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }
    linear_functional_config = {
        "pattern": torch.nn.functional.linear,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    linear_relu_fused_config = {
        "pattern": nni.LinearReLU,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
        "qat_module": nniqat.LinearReLU,
    }
    linear_relu_qat_config = {
        "pattern": nniqat.LinearReLU,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }
    linear_relu_mm_config = {
        "pattern": (torch.nn.ReLU, torch.nn.Linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "fuser_method": reverse_sequential_wrapper2(nni.LinearReLU),
    }
    linear_relu_mf_config = {
        "pattern": (torch.nn.functional.relu, torch.nn.Linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "fuser_method": reverse_sequential_wrapper2(nni.LinearReLU),
    }
    linear_relu_fm_config = {
        "pattern": (torch.nn.ReLU, torch.nn.functional.linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    linear_relu_ff_config = {
        "pattern": (torch.nn.functional.relu, torch.nn.functional.linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    linear_bn_fused_config = {
        "pattern": nni.LinearBn1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
        "qat_module": nniqat.LinearBn1d,
    }
    linear_bn_qat_config = {
        "pattern": nniqat.LinearBn1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }

    conv1d_module_config = {
        "pattern": torch.nn.Conv1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv1d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv1d,
    }
    conv2d_module_config = {
        "pattern": torch.nn.Conv2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv2d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv2d,
        "qat_module": nnqat.Conv2d,
    }
    conv3d_module_config = {
        "pattern": torch.nn.Conv3d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv3d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv3d,
        "qat_module": nnqat.Conv3d,
    }
    # TODO: add support for qat.Conv1d
    conv2d_qat_config = {
        "pattern": nnqat.Conv2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv2d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv2d,
    }
    conv3d_qat_config = {
        "pattern": nnqat.Conv3d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv3d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv3d,
    }
    conv1d_functional_config = {
        "pattern": torch.nn.functional.conv1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    conv2d_functional_config = {
        "pattern": torch.nn.functional.conv2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    conv3d_functional_config = {
        "pattern": torch.nn.functional.conv3d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    conv1d_relu_fused_config = {
        "pattern": nni.ConvReLU1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv1d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv1d,
    }
    conv2d_relu_fused_config = {
        "pattern": nni.ConvReLU2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv2d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv2d,
        "qat_module": nniqat.ConvReLU2d,
    }
    conv3d_relu_fused_config = {
        "pattern": nni.ConvReLU3d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv2d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv3d,
        "qat_module": nniqat.ConvReLU3d,
    }
    # TODO: add support for qat.ConvReLU1d
    conv2d_relu_qat_config = {
        "pattern": nniqat.ConvReLU2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv2d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv2d,
    }
    conv3d_relu_qat_config = {
        "pattern": nniqat.ConvReLU3d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv3d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv3d,
    }
    conv1d_relu_fm_config = {
        "pattern": (torch.nn.ReLU, torch.nn.functional.conv3d),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    conv2d_relu_fm_config = {
        "pattern": (torch.nn.ReLU, torch.nn.functional.conv3d),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    conv3d_relu_fm_config = {
        "pattern": (torch.nn.ReLU, torch.nn.functional.conv3d),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    conv1d_relu_ff_config = {
        "pattern": (torch.nn.functional.relu, torch.nn.functional.conv1d),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    conv2d_relu_ff_config = {
        "pattern": (torch.nn.functional.relu, torch.nn.functional.conv2d),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    conv3d_relu_ff_config = {
        "pattern": (torch.nn.functional.relu, torch.nn.functional.conv3d),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    conv1d_bn_qat_config = {
        "pattern": nniqat.ConvBn1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv1d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv1d,
    }
    conv2d_bn_qat_config = {
        "pattern": nniqat.ConvBn2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv2d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv2d,
    }
    conv3d_bn_qat_config = {
        "pattern": nniqat.ConvBn3d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv3d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv3d,
    }
    conv1d_bn_relu_qat_config = {
        "pattern": nniqat.ConvBnReLU1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv1d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv1d,
    }
    conv2d_bn_relu_qat_config = {
        "pattern": nniqat.ConvBnReLU2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv2d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv2d,
    }
    conv3d_bn_relu_qat_config = {
        "pattern": nniqat.ConvBnReLU3d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Conv3d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv3d,
    }
    return {
        # optional
        "name": "native",
        "configs": [
            linear_module_config,
            linear_qat_config,
            linear_functional_config,
            linear_relu_fused_config,
            linear_relu_qat_config,
            linear_relu_mm_config,
            linear_relu_mf_config,
            #linear_relu_fm_config,
            #linear_relu_ff_config,
            linear_bn_fused_config,
            linear_bn_qat_config,
            conv1d_module_config,
            conv2d_module_config,
            conv3d_module_config,
            conv2d_qat_config,
            conv3d_qat_config,
            conv1d_functional_config,
            conv2d_functional_config,
            conv3d_functional_config,
            conv1d_relu_fused_config,
            conv2d_relu_fused_config,
            conv3d_relu_fused_config,
            conv2d_relu_qat_config,
            conv3d_relu_qat_config,
            #conv1d_relu_fm_config,
            #conv2d_relu_fm_config,
            #conv3d_relu_fm_config,
            #conv1d_relu_ff_config,
            #conv2d_relu_ff_config,
            #conv3d_relu_ff_config,
            conv1d_bn_qat_config,
            conv2d_bn_qat_config,
            conv3d_bn_qat_config,
            conv1d_bn_relu_qat_config,
            conv2d_bn_relu_qat_config,
            conv3d_bn_relu_qat_config,
        ],
    }
