import hashlib
import os.path as osp
import pickle
import shutil

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, download_url
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import numpy as np

import sys
import warnings
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data.dataset import _repr


class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(self, root='datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'peptides-functional')

        self.url = 'https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1'
        # MD5 hash of the intended dataset file
        self.version = '701eb743e899f4d793f0e13c8fa5a1b4'
        self.url_stratified_split = 'https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1'
        self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'peptide_multi_class_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print('Stop download.')
            exit(-1)

    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"sure to delete '{self.processed_dir}' first")

        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in the "
                "pre-processed version of this dataset. If you want to make "
                "use of another pre-fitering technique, make sure to delete "
                "'{self.processed_dir}' first")

        print('Processing...', file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        print('Done!', file=sys.stderr)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir,
                                       'peptide_multi_class_dataset.csv.gz'))
        smiles_list = data_df['smiles']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            # #### Coarsening start ####

            # r = 0.3 

            # node_feat = graph['node_feat']
            # edge_index = graph['edge_index']
            # edge_attr = graph['edge_feat']
            
            # num_edge = len(edge_index.T)
            # edge_index = edge_index.T
            # new_node = np.max(edge_index)
            # coarsened_edge = np.sort(np.random.choice(range(0, num_edge), int(num_edge * r/2), replace = False))
            # mark_del_node_feat = np.full(node_feat.shape[1], -1)
            # mark_del_edge_index = np.full(edge_index.shape[1], -1)
            # mark_del_edge_feat = np.full(edge_attr.shape[1], -1)  

            # for idx_c_edge in range(len(coarsened_edge)):
   
            #     dup_edge = np.array([edge_index[coarsened_edge[idx_c_edge]][1], edge_index[coarsened_edge[idx_c_edge]][0]])
                
            #     for j in range(len(coarsened_edge)):
            #         if j != idx_c_edge:
                            
            #             if (edge_index[coarsened_edge[j]] == dup_edge).all():
            #                 coarsened_edge = np.delete(coarsened_edge, j)
            #                 break

            #     if idx_c_edge+1 == len(coarsened_edge):
            #         break

            # for idx_c_edge in coarsened_edge:
  
            #     new_node += 1

            #     c_source_node = edge_index[idx_c_edge, 0]
            #     c_target_node = edge_index[idx_c_edge, 1]

            #     # print("target edge : ", [c_source_node, c_target_node])
            #     if c_source_node == -1 or c_target_node == -1:
            #         continue

            #     new_node_feat = node_feat[c_source_node] + node_feat[c_target_node]      

            #     # Marking deleted node features by -1 array
            #     node_feat[c_source_node] = mark_del_node_feat
            #     node_feat[c_target_node] = mark_del_node_feat
            #     node_feat = np.vstack((node_feat, new_node_feat))
                
            #     # Replacing coarsened node index to new node index
            #     mask_source = np.logical_or(edge_index[:, 0] == c_source_node, edge_index[:, 0] == c_target_node)
            #     mask_target = np.logical_or(edge_index[:, 1] == c_source_node, edge_index[:, 1] == c_target_node)
            #     edge_index[mask_source, 0] = new_node
            #     edge_index[mask_target, 1] = new_node

            #     # Updating edge features for coarsened edges
            #     edge_attr[mask_source] = edge_attr[mask_source].sum(axis = 0)
            #     edge_attr[mask_target] = edge_attr[mask_target].sum(axis = 0)

            #     # Marking deleted edge indices and features by [-1, -1] and [-1, -1, ... , -1]
            #     mask_self_loop = edge_index[:, 0] == edge_index[:, 1]
            #     edge_index[mask_self_loop] = mark_del_edge_index
            #     edge_attr[mask_self_loop] = mark_del_edge_feat

            # # Node feature part

            # rows_to_delete = np.where(np.all(node_feat == mark_del_node_feat, axis=1)) 
            # node_feat = np.delete(node_feat, rows_to_delete, axis=0) # Delete [-1, ... , -1]

            # # Edge feature part

            # edge_dict = {}
            # for i, edge in enumerate(edge_index):
            #     key = tuple(edge)
            #     if key in edge_dict:
            #         if np.all(edge_attr[edge_dict[key]] == mark_del_edge_feat):
            #             continue
            #         else:
            #             edge_attr[edge_dict[key]] += edge_attr[i]
            #     else:
            #         edge_dict[key] = i
            # unique_edge_attr = edge_attr[list(edge_dict.values())] # Aggregate edge features
            # edge_attr = unique_edge_attr[~np.all(unique_edge_attr == mark_del_edge_feat, axis = 1)] # Delete [-1, ... -1]

            # rows_to_delete = np.where(np.all(edge_index == mark_del_edge_index, axis=1))
            # edge_index = np.delete(edge_index, rows_to_delete, axis=0) # Delete [-1, -1]
            # edge_index = np.unique(edge_index, axis = 0).T # Delete duplicated node pairs

            # flattened_edge_index = edge_index.flatten()
            # unique_elements = np.unique(flattened_edge_index)
            # element_mapping = {element: new_value for new_value, element in enumerate(unique_elements)}
            # edge_index = np.vectorize(element_mapping.get)(edge_index)

            # graph['num_nodes'] = len(node_feat)
            # graph['node_feat'] = node_feat
            # graph['edge_index'] = edge_index
            # graph['edge_feat'] = edge_attr

            # #### Coarsening finish ####

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(
                torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([eval(data_df['labels'].iloc[i])])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root,
                              "splits_random_stratified_peptide.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict

if __name__ == '__main__':
    dataset = PeptidesFunctionalDataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())
