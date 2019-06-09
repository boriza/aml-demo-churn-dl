#%% [markdown]
# # Churn Prediction
# 
# This notebook will introduce the use of the churn dataset to create churn prediction model using deep kernel learning.
# 
# The dataset used to ingest is from SIDKDD 2009 competition. 
# 
# The pipeline is composed using Azure ML pipeline and trained on Azure ML compute with hyper parameters of the gaussian process and the neural network jointly tuned through hyperdrive.

#%%
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os
import urllib

from azureml.core import  (Workspace,Run,VERSION,
                           Experiment,Datastore)
from azureml.core.runconfig import (RunConfiguration,
                                    DEFAULT_GPU_IMAGE)
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import (AmlCompute, ComputeTarget)
from azureml.exceptions import ComputeTargetException
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import (Pipeline, 
                                   PipelineData)
from azureml.pipeline.steps import (HyperDriveStep,PythonScriptStep)
from azureml.train.dnn import PyTorch
from azureml.train.hyperdrive import *
from azureml.widgets import RunDetails


print('SDK verison', VERSION)

#%% [markdown]
# ## Variables declaration
# 
# Declare variables to be used through out, please fill in the Azure subscription ID, resource-group and workspace name to connect to your Azure ML workspace.

#%%
# SUBSCRIPTION_ID = ''
# RESOURCE_GROUP = ''
# WORKSPACE_NAME = ''
#Instead download AML config.json from azure portal 'Machine Learning service workspace'

PROJECT_DIR = os.getcwd()
EXPERIMENT_NAME = "customer_churn"
CLUSTER_NAME = "gpu-cluster"
DATA_DIR = os.path.join(PROJECT_DIR,'data')
TRAIN_DIR = os.path.join(PROJECT_DIR,'train')
PREPROCESS_DIR = os.path.join(PROJECT_DIR,'preprocess')

SOURCE_URL ='https://amlgitsamples.blob.core.windows.net/churn'
FILE_NAME = 'CATelcoCustomerChurnTrainingSample.csv'

#%% [markdown]
# ## Initialize workspace
# 
# Initialize a workspace object 

#%%
ws = Workspace.from_config()
print('Workspace loaded:', ws.name)

#%% [markdown]
# ## Data download
# 
# Download Dataset locally to experiment folder

#%%
os.makedirs(DATA_DIR, exist_ok=True)

urllib.request.urlretrieve(os.path.join(SOURCE_URL,FILE_NAME), 
                           filename = os.path.join(DATA_DIR,FILE_NAME))

#%% [markdown]
# ## Upload  dataset to blob datastore
# 
# Upload dataset to workspace default blob storage which will be mounted on AzureML compute during pipeline execution.

#%%
default_store = default_datastore=ws.datastores["workspaceblobstore"]
default_store.upload(src_dir=DATA_DIR, target_path='churn', overwrite=True, show_progress=True)

#%% [markdown]
# ## Retrieve or create a Azure Machine Learning compute
# 
# Here we create a new Azure Machine Learning Compute in the current workspace, if it doesn't already exist. We will then run the training script on this compute target.
# 
# If you have already created an Azure ML compute in your workspace, just provide it's name in the cell below to have it used for Azure ML pipeline execution.

#%%
cluster_name = "gpu-cluster"

try:
    cluster = ComputeTarget(ws, cluster_name)
    print(cluster_name, "found")
    
except ComputeTargetException:
    print(cluster_name, "not found, provisioning....")
    provisioning_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',max_nodes=1)

    
    cluster = ComputeTarget.create(ws, cluster_name, provisioning_config)
    
cluster.wait_for_completion(show_output=True)

#%% [markdown]
# ## Pipeline definition
# 
# 
# The Azure ML pipeline is composed of two steps: 
#  
#  - Data pre-processing which consist of one-hot encoding categorical features, normalization of the features set, spliting of dataset into training/testing sets and finally writing out the output to storage.
#  
#  - Hyperdrive step that tune and train the deep kernel learning model using GPytorch and Pytorch estimator 
#%% [markdown]
# ## Pipeline data input/output
# 
# Here, we define the input and intermediary dataset that will be used by the pipeline steps.

#%%
input_dir = DataReference(datastore=default_store,
                          data_reference_name="input_data",
                          path_on_datastore="churn"
                         )

processed_dir = PipelineData(name = 'processed_data',
                             datastore=default_store
                            )

#%% [markdown]
# ## Pipeline 1st step: Data Preprocessing
# 
# We start by defining the run configuration with the needed dependencies by the preprocessing step.
# 
# In the cell that follow, we compose the first step of the pipeline.
# 

#%%
cd = CondaDependencies()
cd.add_conda_package('pandas')
cd.add_conda_package('matplotlib')
cd.add_conda_package('numpy')
cd.add_conda_package('scikit-learn')


run_config = RunConfiguration(framework="python",
                              conda_dependencies= cd)
run_config.target = cluster
run_config.environment.docker.enabled = True
run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
run_config.environment.python.user_managed_dependencies = False


#%%
pre_processing = PythonScriptStep(
                            name='preprocess dataset',
                            script_name='preprocess.py',
                            arguments=['--input_path', input_dir,\
                                         '--output_path', processed_dir],
                            inputs=[input_dir],
                            outputs=[processed_dir],
                            compute_target=cluster_name,
                            runconfig=run_config,
                            source_directory=PREPROCESS_DIR
                        )

#%% [markdown]
# ## Pipeline second step: training
# 
# For the second step, we start by defining the pytorch estimator that will be used to traing the Stochastic variational deep kernel learning model using Gpytorch.

#%%
estimator = PyTorch(source_directory=TRAIN_DIR,
                    conda_packages=['pandas', 'numpy', 'scikit-learn'],
                    pip_packages=['gpytorch'],
                    compute_target=cluster,
                    entry_script='svdkl_entry.py',
                    use_gpu=True)

#%% [markdown]
# Here, we configure Hyperdrive by defining the hyperparametes space and select choose Area under the curve as the metric to optimize for.

#%%
ps = RandomParameterSampling(
    {
        '--batch-size': choice(4096,8192),
        '--epochs': choice(500),
        '--neural-net-lr': loguniform(-4,-2),
        '--likelihood-lr': loguniform(-4,-2),
        '--grid-size': choice(32,64),
        '--grid-bounds': choice(-1,0),
        '--latent-dim': choice(2),
        '--num-mixtures': choice(4,6,8)
    }
)

early_termination_policy = BanditPolicy(evaluation_interval=10, slack_factor=0.1)

hd_config = HyperDriveRunConfig(estimator=estimator, 
                                hyperparameter_sampling=ps,
                                policy=early_termination_policy,
                                primary_metric_name='auc', 
                                primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                                max_total_runs=20,
                                max_concurrent_runs=4)

#%% [markdown]
# Last, we define the hyperdrive step of the pipeline.

#%%
metrics_output_name = 'metrics_output'
metirics_data = PipelineData(name='metrics_data',
                             datastore=default_store,
                             pipeline_output_name=metrics_output_name)

hd_step = HyperDriveStep(
    name="hyper parameters tunning",
    hyperdrive_config=hd_config,
    estimator_entry_script_arguments=['--data-folder', processed_dir],
    inputs=[processed_dir],
    metrics_output=metirics_data)

#%% [markdown]
# ## Build & Execute pipeline

#%%
pipeline = Pipeline(workspace=ws, steps=[hd_step],
                    default_datastore=default_store
                   )
#Run the pipeline 
#%% [markdown]     
pipeline_run = Experiment(ws, 'Customer_churn').submit(pipeline,
                                                      regenerate_outputs=True)


#%%
RunDetails(pipeline_run).show()


