import pandas as pd
from functools import lru_cache

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

print('IMPORT UTILS')
def list_models(api_key):
  metadata = (('authorization', 'Key ' + api_key),) 

  list_models_response = stub.ListModels(
    service_pb2.ListModelsRequest(),
    metadata=metadata
  )

  return [model.id for model in list_models_response.models]


def get_all_preds_and_urls(api_key, model_id):
  metadata = (('authorization', 'Key ' + api_key),)

  inputs = {}
  outputs = {}
  page = 0
  while True:
      retrieved_inputs = stub.ListInputs(
          service_pb2.ListInputsRequest(page=page, per_page=128),
          metadata=metadata
      ).inputs
      if len(retrieved_inputs) == 0:
          break
      
      for app_input in retrieved_inputs:
          inputs[app_input.id] = app_input
          
      page += 1
      
      retrieved_outputs = stub.PostModelOutputs(
          service_pb2.PostModelOutputsRequest(model_id=model_id, inputs=retrieved_inputs),
          metadata=metadata
      ).outputs
      for output in retrieved_outputs:
          outputs[output.input.id] = output.data
    
  urls = {}
  input_ids = []
  ground_truths = []
  pred = []
  confidence = []
  for input_id, output_list in outputs.items():
      urls[input_id] = inputs[input_id].data.image.url
      for output_concept in output_list.concepts:
          input_ids.append(input_id)
          ground_truths.append(inputs[input_id].data.concepts[0].name)
          pred.append(output_concept.name)
          confidence.append(output_concept.value)
  preds_df = pd.DataFrame({'input_id': input_ids, 'ground_truth': ground_truths, 'pred': pred, 'confidence': confidence})

  return preds_df, urls

def get_least_conf_inputs(preds_df, thresh=1e-10):
  most_conf_preds_list = []
  for input_id, rows in preds_df.groupby('input_id'):
    most_conf_preds_list.append(rows.sort_values('confidence', ascending=False).iloc[0])
  most_conf_preds = pd.DataFrame(most_conf_preds_list)
  most_conf_preds = most_conf_preds[most_conf_preds['confidence'] > thresh]
  least_conf_inputs = most_conf_preds.sort_values('confidence')['input_id'].values
  return least_conf_inputs

@lru_cache(maxsize=1)
def pred_and_create_dfs(api_key, model_id):
    preds_df, urls = get_all_preds_and_urls(api_key, model_id)
    least_conf_input_ids = get_least_conf_inputs(preds_df)
    return preds_df, urls, least_conf_input_ids

