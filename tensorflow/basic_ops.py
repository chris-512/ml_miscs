#!/usr/bin/env python3

import tensorflow as tf

tf.square
tf.reduce_sum
tf.sqrt
tf.concat
tf.losses.compute_weighted_loss
tf.losses.get_total_loss

slim = tf.contrib.slim
tf.decode_raw

tf.abs()
tf.matmul()
tf.multiply()
tf.multinomial()
tf.make_ndarray()
tf.map_fn()
tf.map_stage()
tf.map_size()
tf.matrix_transpose()
tf.transpose()
tf.add_n()
tf.add_to_collection()
tf.all_variables()
tf.arg_max()
tf.arg_min()
tf.assert_equal()
tf.assert_greater()
tf.assert_greater_equal()
tf.assert_integer()
tf.assert_less()
tf.assert_less_equal()
tf.assert_non_negative()
tf.assign()
tf.assign_add()
tf.assign_sub()
tf.argmax()
tf.argmin()
tf.clip_by_average_norm()
tf.cast()
tf.case()
tf.ceil()
tf.check_numerics()
tf.check_ops
tf.cholesky()
tf.cholesky_grad()
tf.cholesky_solve()
tf.clip_by_global_norm()
tf.clip_by_average_norm()
tf.clip_by_norm()
tf.clip_by_value()
tf.colocate_with()
tf.complex()
tf.cond
tf.confusion_matrix()
tf.conj()
tf.cross()
tf.cumprod()
tf.cumsum()

tf.constant()
tf.convert_to_tensor()
tf.convert_to_tensor_or_indexed_slices()
tf.convert_to_tensor_or_sparse_tensor()
tf.decode_base64()
tf.decode_csv()
tf.decode_json_example()
tf.decode_raw()
tf.device()
tf.diag()
tf.diag_part()
tf.div()
tf.divide()
tf.batch_to_space_nd()
tf.space_to_batch_nd()
tf.batch_to_space()
tf.space_to_batch()

tf.depth_to_space()
tf.space_to_depth()

tf.dtypes

tf.get_collection()
tf.get_collection_ref()
tf.get_default_session()
tf.get_local_variable
tf.get_seed()
tf.get_session_handle()
tf.get_session_tensor()
tf.get_default_graph()
tf.get_summary_op()
tf.get_variable()
tf.get_variable_scope()
tf.set_random_seed()
tf.serialize_tensor()
tf.save_v2()
tf.scalar_mul()
tf.scan()
tf.scatter_add()
tf.scatter_div()
tf.scatter_mul()
tf.scatter_nd()
tf.scatter_nd_add()
tf.scatter_nd_non_aliasing_add()
tf.scatter_nd_sub()
tf.scatter_nd_update()
tf.scatter

tf.tables_initializer()
tf.tensordot()
tf.tf_logging
tf.tile()
tf.to_bfloat16()
tf.to_double()
tf.to_float()
tf.to_int32()
tf.to_int64()

tf.trace()
tf.trainable_variables()
tf.transpose()
tf.truncated_normal()
tf.truediv()
tf.sparse_transpose()
tf.sparse_tensor_dense_matmul()
tf.sparse_accumulator_apply_gradient()
tf.sparse_accumulator_take_gradient()
tf.sparse_add()
tf.sparse_concat()
tf.sparse_conditional_accumulator()
tf.sparse_mask()
tf.sparse_matmul()
tf.sparse_maximum()
tf.sparse_merge()
tf.sparse_minimum()

tf.sparse_reduce_max()
tf.sparse_reduce_max_sparse()

tf.reduce_all()
tf.reduce_any()
tf.reduce_join()
tf.reduce_logsumexp()
tf.reduce_max()
tf.reduce_mean()
tf.reduce_min()
tf.reduce_prod()
tf.reduce_sum()
tf.reduced_shape()

tf.random_crop()
tf.random_gamma()
tf.random_normal()
tf.random_poisson()
tf.random_poisson_v2()
tf.random_shuffle()
tf.random_uniform()

tf.where()
tf.while_loop()
tf.write_file()
tf.read_file()

tf.record_input()
tf.reshape()
tf.restore_v2()
tf.reverse()
tf.ordered_map_clear()
tf.ordered_map_incomplete_size()
tf.ordered_map_peek()
tf.ordered_map_size()
tf.ordered_map_stage()
tf.ordered_map_unstage()
tf.ordered_map_unstage_no_key()

tf.matrix_diag()


tf.negative()
tf.norm()
tf.is_nan()
tf.is_finite()
tf.is_inf()
tf.is_non_decreasing()
tf.is_numeric_tensor()
tf.is_strictly_increasing()
tf.is_variable_initialized

tf.global_variables_initializer()
tf.global_variables()
tf.global_norm()

tf.local_variables()
tf.local_variables_initializer()
tf.get_local_variable
tf.initialize_local_variables

tf.equal()
tf.einsum()
tf.extract_image_patches()

tf.make_all()

# tf.nn

tf.nn.atrous_conv2d()
tf.nn.atrous_conv2d_transpose()
tf.nn.avg_pool()
tf.nn.avg_pool3d()
tf.nn.batch_norm_with_global_normalization()
tf.nn.batch_normalization()
tf.nn.bias_add()
tf.nn.bias_add_grad()
tf.nn.bias_add_v1()
tf.nn.bidirectional_dynamic_rnn()
tf.nn.xw_plus_b()


tf.nn.conv1d()
tf.nn.conv2d()
tf.nn.conv2d_backprop_input()
tf.nn.conv2d_backprop_filter()
tf.nn.conv2d_transpose()
tf.nn.conv3d()
tf.nn.conv3d_backprop_filter()
tf.nn.conv3d_backprop_filter_v2()
tf.nn.conv3d_backprop_input()
tf.nn.conv3d_backprop_input_v2()
tf.nn.conv3d_transpose()

tf.nn.depthwise_conv2d()
tf.nn.depthwise_conv2d_native()
tf.nn.depthwise_conv2d_native_backprop_filter()
tf.nn.depthwise_conv2d_native_backprop_input()

tf.nn.fused_pad_conv2d()
tf.nn.fused_batch_norm()
tf.nn.fused_batch_norm_grad()
tf.nn.fused_batch_norm_grad_v2()
tf.nn.fused_resize_and_pad_conv2d()

tf.nn.convolution()
tf.nn.quantized_conv2d()
tf.nn.separable_conv2d()

# pooling operations
# (1) average pooling
tf.nn.avg_pool()
tf.nn.avg_pool3d()
tf.nn.quantized_avg_pool()
tf.nn.fractional_avg_pool()

# (2) max pooling
tf.nn.max_pool()
tf.nn.max_pool3d()
tf.nn.max_pool_grad_grad_v2()
tf.nn.max_pool_grad_v2()
tf.nn.max_pool_with_argmax()
tf.nn.fractional_max_pool()
tf.nn.quantized_max_pool()

tf.nn.l2_loss()
tf.nn.nce_loss()
tf.nn.ctc_loss()
tf.nn.sampled_softmax_loss()
tf.nn.log_poisson_loss()

tf.nn.log_softmax()
tf.nn.log_uniform_candidate_sampler()
tf.nn.sigmoid_cross_entropy_with_logits()
tf.nn.softmax_cross_entropy_with_logits()
tf.nn.sparse_softmax_cross_entropy_with_logits()
tf.nn.weighted_cross_entropy_with_logits()

tf.nn.relu()
tf.nn.relu6()
tf.nn.relu_layer()
tf.nn.leaky_relu()
tf.nn.elu()
tf.nn.crelu()
tf.nn.quantized_relu()
tf.nn.selu()
tf.nn.quantized_relu6()
tf.nn.quantized_relu_x()

tf.nn.sigmoid()
tf.nn.softsign()
tf.nn.softmax()
tf.nn.softplus()

tf.nn.dilation2d()
tf.nn.dilation2d_backprop_filter()
tf.nn.dilation2d_backprop_input()
tf.nn.dropout()

# rnn cell
tf.nn.bidirectional_dynamic_rnn()
tf.nn.raw_rnn()
tf.nn.static_rnn()
tf.nn.static_bidirectional_rnn()
tf.nn.dynamic_rnn()
tf.nn.static_state_saving_rnn()

tf.nn.embedding_lookup()
tf.nn.embedding_lookup_sparse()

tf.nn.ctc_beam_search_decoder()
tf.nn.ctc_greedy_decoder()

tf.nn.erosion2d()
tf.nn.zero_fraction()

tf.nn.l2_normalize()

# math ops
tf.nn.math_ops

tf.nn.array_ops
tf.nn.clip_ops
tf.nn.control_flow_ops
tf.nn.embedding_ops
tf.nn.embedding_ops
tf.nn.nn_ops
tf.nn.sparse_ops
tf.nn.tensor_array_ops

tf.nn.top_k()
tf.nn.in_top_k()

tf.nn.weighted_moments()
tf.nn.with_space_to_batch()

# slim

tf.FixedLenFeature
tf.FixedLenSequenceFeature
tf.VarLenFeature
tf.Variable
tf.VariableScope
tf.VarianceScaling
tf.PartitionedVariable
tf.SparseConditionalAccumulator
tf.SparseFeature
tf.SparseTensor
tf.Assert
tf.Barrier
tf.BaseStagingArea
tf.ConditionalAccumulatorBase
tf.QueueBase
tf.ReaderBase

tf.RandomNormal
tf.RandomShuffleQueue
tf.RandomUniform

tf.RecordInput
tf.RegisterGradient
tf.RegisterShape
tf.RunMetadata
tf.RunOptions


# Reader
tf.FixedLengthRecordReader
tf.IdentityReader
tf.TextLineReader
tf.TFRecordReader
tf.WholeFileReader
tf.LMDBReader
tf.FixedLengthRecordReader

tf.Graph.get_collection_ref()
tf.Graph.get_collection()
tf.Graph.get_all_collection_keys()
tf.Graph.get_name_scope()
tf.Graph.get_operation_by_name()
tf.Graph.get_operations()
tf.Graph.get_tensor_by_name()

tf.Graph.as_default()
tf.Graph.as_graph_def()
tf.Graph.as_graph_element()

tf.GraphKeys.ACTIVATIONS
tf.GraphKeys.ASSET_FILEPATHS
tf.GraphKeys.BIASES
tf.GraphKeys.CONCATENATED_VARIABLES
tf.GraphKeys.COND_CONTEXT
tf.GraphKeys.EVAL_STEP
tf.GraphKeys.GLOBAL_STEP
tf.GraphKeys.GLOBAL_VARIABLES
tf.GraphKeys.ASSET_FILEPATHS
tf.GraphKeys.LOCAL_RESOURCES
tf.GraphKeys.LOSSES
tf.GraphKeys.LOCAL_VARIABLES
tf.GraphKeys.MOVING_AVERAGE_VARIABLES
tf.GraphKeys.QUEUE_RUNNERS

tf.GraphKeys.MODEL_VARIABLES
tf.GraphKeys.REGULARIZATION_LOSSES
tf.GraphKeys.RESOURCES
tf.GraphKeys.SAVEABLE_OBJECTS
tf.GraphKeys.SAVERS
tf.GraphKeys.SUMMARIES

tf.GraphKeys.UPDATE_OPS
tf.GraphKeys.TRAIN_OP
tf.GraphKeys.OP
tf.GraphKeys.SUMMARY_OP
tf.GraphKeys.READY_OP
tf.GraphKeys.READY_FOR_LOCAL_INIT_OP
tf.GraphKeys.INIT_OP
tf.GraphKeys.LOCAL_INIT_OP

tf.GraphKeys.TRAINABLE_VARIABLES
tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES
tf.GraphKeys.WEIGHTS

tf.GPUOptions
tf.GraphDef
tf.MetaGraphDef
tf.NoGradient
tf.SessionLog

# session
tf.Session.as_default()
tf.Session.close()
tf.Session.list_devices()
tf.Session.make_callable()
tf.Session.partial_run()
tf.Session.partial_run_setup()
tf.Session.reset()
tf.Session.run()

tf.InteractiveSession.run()
tf.InteractiveSession.partial_run_setup()
tf.InteractiveSession.partial_run()
tf.InteractiveSession.list_devices()
tf.InteractiveSession.close()
tf.InteractiveSession.as_default()

tf.zeros_like()
tf.zeros()
tf.ones_like()
tf.ones()

# initializer
tf.ones_initializer
tf.zeros_initializer
tf.initialize_local_variables
tf.initialize_all_tables
tf.initialize_all_variables
tf.local_variables_initializer()
tf.global_variables_initializer()
tf.constant_initializer
tf.variables_initializer()

tf.eye()
tf.expand_dims()
tf.random_shuffle()
tf.expm1()
tf.as_dtype()
tf.as_string()

# slice
tf.slice()
tf.sparse_slice()
tf.strided_slice()
tf.convert_to_tensor_or_indexed_slices()
tf.resource_strided_slice_assign()
tf.strided_slice_assign()
tf.strided_slice_grad()

tf.gather()
tf.gather_nd()
tf.gather_v2()
tf.get_summary_op()
tf.gradients()
tf.boolean_mask()
tf.sparse_mask()
tf.sequence_mask()

tf.random_gamma()
tf.digamma()
tf.igamma()
tf.lgamma()
tf.polygamma()
tf.igammac()

tf.tensor_shape.as_shape()



# gfile
tf.gfile.Copy()
tf.gfile.DeleteRecursively()
tf.gfile.Exists()
tf.gfile.Glob()
tf.gfile.IsDirectory()
tf.gfile.ListDirectory()
tf.gfile.MakeDirs()
tf.gfile.MkDir()
tf.gfile.Remove()
tf.gfile.Rename()
tf.gfile.Stat()
tf.gfile.Walk()

tf.gfile.GFile.close()
tf.gfile.GFile.flush()
tf.gfile.GFile.read()
tf.gfile.GFile.readline()
tf.gfile.GFile.readlines()
tf.gfile.GFile.seek()
tf.gfile.GFile.tell()
tf.gfile.GFile.write()

tf.gfile.FastGFile.close()
tf.gfile.FastGFile.write()
tf.gfile.FastGFile.tell()
tf.gfile.FastGFile.seek
tf.gfile.FastGFile.readlines()
tf.gfile.FastGFile.readline()
tf.gfile.FastGFile.read()
tf.gfile.FastGFile.flush()

tf.pad()
tf.range()

tf.tensor_array_ops
tf.sparse_ops
tf.control_flow_ops
tf.clip_ops
tf.array_ops
tf.check_ops
tf.functional_ops


tf.functional_ops.scan()
tf.functional_ops.map_fn()
tf.functional_ops.foldl()
tf.functional_ops.foldr()
tf.functional_ops.remote_call()

