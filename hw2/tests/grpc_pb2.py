# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: grpc.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ngrpc.proto\"I\n\x0fModelAddRequest\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x18\n\x04type\x18\x02 \x02(\x0e\x32\n.ModelType\x12\x0e\n\x06params\x18\x03 \x02(\t\"\"\n\x12ModelSimpleRequest\x12\x0c\n\x04name\x18\x01 \x02(\t\"\xf2\x01\n\x13ModelPredictRequest\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x15\n\rfixed_acidity\x18\x02 \x02(\x02\x12\x18\n\x10volatile_acidity\x18\x03 \x02(\x02\x12\x13\n\x0b\x63itric_acid\x18\x04 \x02(\x02\x12\x16\n\x0eresidual_sugar\x18\x05 \x02(\x02\x12\x11\n\tchlorides\x18\x06 \x02(\x02\x12\x1b\n\x13\x66ree_sulfur_dioxide\x18\x07 \x02(\x02\x12\x0f\n\x07\x64\x65nsity\x18\x08 \x02(\x02\x12\n\n\x02pH\x18\t \x02(\x02\x12\x11\n\tsulphates\x18\n \x02(\x02\x12\x0f\n\x07\x61lcohol\x18\x0b \x02(\x02\"\x07\n\x05\x45mpty\"I\n\x13ModelSimpleResponse\x12!\n\x06status\x18\x01 \x02(\x0e\x32\x11.ModelMessageType\x12\x0f\n\x07message\x18\x02 \x02(\t\"&\n\x14ModelPredictResponse\x12\x0e\n\x06result\x18\x01 \x02(\x02\"6\n\x11ModelListResponse\x12!\n\x06models\x18\x01 \x03(\x0b\x32\x11.ModelDescription\"\x82\x01\n\x10ModelDescription\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x18\n\x04type\x18\x02 \x02(\x0e\x32\n.ModelType\x12\x11\n\tisTrained\x18\x03 \x02(\x08\x12\x16\n\x0etrain_accuracy\x18\x04 \x01(\x02\x12\x1b\n\x13validation_accuracy\x18\x05 \x01(\x02*$\n\tModelType\x12\n\n\x06logreg\x10\x00\x12\x0b\n\x07xgboost\x10\x01*&\n\x10ModelMessageType\x12\x06\n\x02OK\x10\x00\x12\n\n\x06\x46\x61iled\x10\x01\x32\x9d\x02\n\x05Model\x12)\n\tModelList\x12\x06.Empty\x1a\x12.ModelListResponse\"\x00\x12\x33\n\x08ModelAdd\x12\x10.ModelAddRequest\x1a\x13.ModelSimpleRequest\"\x00\x12:\n\x0bModelRemove\x12\x13.ModelSimpleRequest\x1a\x14.ModelSimpleResponse\"\x00\x12\x39\n\nModelTrain\x12\x13.ModelSimpleRequest\x1a\x14.ModelSimpleResponse\"\x00\x12=\n\x0cModelPredict\x12\x14.ModelPredictRequest\x1a\x15.ModelPredictResponse\"\x00')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'grpc_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MODELTYPE._serialized_start=683
  _MODELTYPE._serialized_end=719
  _MODELMESSAGETYPE._serialized_start=721
  _MODELMESSAGETYPE._serialized_end=759
  _MODELADDREQUEST._serialized_start=14
  _MODELADDREQUEST._serialized_end=87
  _MODELSIMPLEREQUEST._serialized_start=89
  _MODELSIMPLEREQUEST._serialized_end=123
  _MODELPREDICTREQUEST._serialized_start=126
  _MODELPREDICTREQUEST._serialized_end=368
  _EMPTY._serialized_start=370
  _EMPTY._serialized_end=377
  _MODELSIMPLERESPONSE._serialized_start=379
  _MODELSIMPLERESPONSE._serialized_end=452
  _MODELPREDICTRESPONSE._serialized_start=454
  _MODELPREDICTRESPONSE._serialized_end=492
  _MODELLISTRESPONSE._serialized_start=494
  _MODELLISTRESPONSE._serialized_end=548
  _MODELDESCRIPTION._serialized_start=551
  _MODELDESCRIPTION._serialized_end=681
  _MODEL._serialized_start=762
  _MODEL._serialized_end=1047
# @@protoc_insertion_point(module_scope)
