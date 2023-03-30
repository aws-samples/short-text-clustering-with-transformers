# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Amazon Software License (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
#
# http://aws.amazon.com/asl/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions
# and limitations under the License.

from __future__ import print_function

import os

import argparse
import csv
import pandas as pd

from sklearn.cluster import SpectralClustering


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--n-init', type=int, default=100)
    parser.add_argument('--affinity', type=str, default='cosine')
    parser.add_argument('--n-neighbors', type=int, default=10)
    parser.add_argument('--assign-labels', type=str, default='kmeans')
    parser.add_argument('--random-state', type=int, default=None)
    
    args = parser.parse_args()
    
    input_data_location = "/opt/ml/processing/input"
    output_data_location = "/opt/ml/processing/output"
    
    print("Args:{}".format(args))
    
    print("Loading embeddings")
    title_embeddings_df = pd.read_csv(os.path.join(input_data_location, 'blog_title_embeddings.csv'))
    
    scikit_clustering_model = SpectralClustering(n_clusters=args.n_clusters, 
                                     n_init=args.n_init, 
                                     affinity=args.affinity, 
                                     n_neighbors=args.n_neighbors, 
                                     assign_labels=args.assign_labels, 
                                     random_state=args.random_state)
    
    print("Performing clustering")
    embeddings = title_embeddings_df[title_embeddings_df.columns[0:-1]] #Take all columns (embedding) except the title column
    title_embeddings_df['cluster_label'] = scikit_clustering_model.fit_predict(embeddings)
    
    #Save the data
    print("Saving clusters!")
    title_embeddings_df.to_csv(os.path.join(output_data_location, "clustered_blog_titles_with_embeddings.csv"), index=False)