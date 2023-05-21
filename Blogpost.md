# Mini-project

![](./images/1200px-DHRS7B_homology_model.png)

## Introduction
In recent years, the field of machine learning has been increasingly applied to the analysis and understanding of 3D structures of large biomolecules. This emerging area holds great promise for advancing our knowledge in fields such as biochemistry, structural biology, and drug discovery (such as finding a COVID vaccin). Despite the growing interest, there is still a lack of a comprehensive network architecture that effectively combines the geometric and relational aspects inherent in this domain.

To bridge this gap, the paper titled "Learning from Protein Structure with Geometric Vector Perceptrons" by Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael J.L. Townshend, and Ron O. Dror from Stanford University introduces a novel approach that addresses the limitations of existing architectures. The authors propose the use of geometric vector perceptrons (GVP's), which extend the functionality of standard dense layers to operate on collections of Euclidean vectors. By integrating these layers into graph neural networks, the proposed approach enables simultaneous geometric and relational reasoning on efficient representations of macromolecules.

The main objective of this study is to demonstrate the effectiveness of GVP's in tackling two fundamental challenges in learning from protein structure: model quality assessment (MQA) and computational protein design (QPD). Through comprehensive experiments and evaluations, the authors compare their approach with existing classes of architectures, including state-of-the-art convolutional neural networks and graph neural networks. The results showcase their state-of-the-art performance of their proposed method in both problem domains, highlighting its potential to advance the field of learning from protein structure.

In this review, we will provide an analysis of the key components of the paper, discussing its strengths, weaknesses, and potential implications. Additionally, we will present our group's response to the paper, highlighting our novel contribution and its relevance to the research presented. Finally, we will delve into the results obtained by our work, linking them to the code available in the accompanying Jupyter notebook. By the end of this blogpost, readers will gain a comprehensive understanding of the significance of the paper's findings and our contributions to the field.

Before proceeding further, it is essential to review related work in the area of learning from protein structure, setting the stage for the novel approach proposed by Jing et al.

### Related work



## Weaknesses/Strengths/Potential
The paper "Learning from Protein Structure with Geometric Vector Perceptrons" presents an innovative approach that addresses the limitations of existing network architectures for learning from protein structure as discussed in the introduction. While the proposed method introduces significant advancements, such as capturing geometric properties in GNNs, it is important to consider its strengths, weaknesses, and potential implications.

### Geometric vector perceptrons explained
The basic idea behind GVPs is to represent protein structures as sets of geometric vectors. Each vector in the set corresponds to a specific geometric feature of the protein, such as the position of an atom or the orientation of a molecular bond. These vectors capture the spatial relationships and characteristics of the protein's constituent parts.

#### Weaknesses
One of the primary weaknesses of the proposed method lies in the use of geometric vector perceptrons (GVP) as the primary building block for the network architecture. Although GVPs offer the advantage of operating on collections of Euclidean vectors, they may not fully capture the complexity and long-range dependencies present in protein structure data. This limitation could potentially hinder the model's ability to learn intricate spatial relationships within protein structures accurately.

Furthermore, the reliance on dense layers within the GVPs introduces computational inefficiencies, especially when dealing with large-scale datasets. The increased parameter count and computational complexity may pose challenges when deploying the model in real-world scenarios or when working with limited computational resources.

#### Strengths
Despite the identified weaknesses, the proposed approach also exhibits several notable strengths. The integration of geometric and relational reasoning within the network architecture allows for a comprehensive understanding of the structural aspects of macromolecules. By leveraging GVPs and graph neural networks, the model can capture both local geometric features and global relational information, providing a more holistic representation of protein structure.

Moreover, the authors demonstrate the effectivenes of their approach by showcasing significant improvements over existing architectures, including state-of-the-art convolutional neural networks and graph neural networks. This performance enhancement is particularly evident in the two fundamental challenges addressed in the paper: model quality assessment and computational protein design. The superior results obtained by the proposed method underscore its potential as a powerful tool for advancing the field of learning from protein structure.

#### Potential
The paper's proposed approach opens up several exciting possibilities for future research and applications. One potential avenue for improvement is the replacement of GVPconv with a transformerConv. Transformers have shown remarkable success in various natural language processing tasks and have the potential to capture long-range dependencies effectively. By incorporating a transformerConv module into the network architecture, it is possible to enhance the model's ability to capture global relationships and improve its performance on complex protein structure data.

Additionally, the authors' idea of replacing the linear layers within the transformerConv with GVPs introduces a novel combination of architectural components. This, approach has the potential to exploit the strengths of both GVPs and transformers, leveraging the geometric reasoning capabilities of GVPs while benefiting from the attention mechanisms and parallelizability of transformers. Further exploration and experimentation with this hybrid architecture could yield even more robust and efficient models for learning from protein structure.


### Methods 
#### Datasets and tasks
In order to assess the quality of our reimplementation of the GVP architecture, as well as the GVP architecture with the added transformer, we used the ATOM3D set of datasets. ATOM3D is a collection of benchmark datasets for machine learning in structural biology. It concerns the three-dimensional structure of biomolecules, including proteins, small molecules, and nucleic acids and is designed as a benchmark for machine learning methods which operate on 3D molecular structure. The different datasets in ATOM3D also include specific tasks for our model to optimise and evaluate. For our research, we used the following sets of dataset and task from ATOM3D.

**RES -**
The RES dataset consists of atomic environments extracted from nonredundant structures in the PDB. This dataset is accompanied by a classification task where the goal is to predict the identity of the amino acid in the center of the environment based on all other atoms in that environment.

**LBA -**
The LBA dataset is uses a curated database containing protein-ligand complexes from the PDB and their corresponding binding strengths. The task at hand is We predict pK = -log(K), where K is the binding affinity in Molar units.

**SMP -**
The SMP dataset uses the QM9 dataset (Ruddigkeit et al., 2012; Ramakrishnan et al., 2014)(**still have to properly cite everything**), which contains structures and energetic, electronic, and thermodynamic properties for 134,000 stable small organic molecules, obtained from quantum-chemical calculations. The task here is to predict all molecular properties from the ground-state structure.

**LEP -**
This is a novel dataset created by curating proteins from several families with both ”active” and ”inactive” state structures, and model in 527 small molecules with known activating or inactivating function using the program Glide. The corresponding task here can be formulated as a binary classification tsk where the goal is to predict whether or not a molecule bound to the structures will be an activator of the protein's function or not.

**MSP -**
This is a novel dataset which was derived by collecting single-point mutations from the SKEMPI database (Jankauskaitė et al., 2019)(**properly cite**) and model each mutation into the structure to produce mutated structures. The task here can be seen as a binary classification task where we predict whether the stability of the complex increases as a result of the mutation.


#### Vector Gating
Equivariant Graph Neural Networks for 3D Macromolecular Structure paper, toont aan dat ie beter is dus hebben vector gating aangezet bij elke experiment




**TODO: cite**

 

## Novel Contribution
Description of Novel contributions of our work.
- Tested the integration of TransformeConv in GVP model on MQA Cong,  - GVPTransformer from https://github.com/congliuUvA/gvp
- Tested with Bert embeddings initialization on sequence prediction task
- Tested with ensemble model with Bert amino acid mask prediction on sequence prediction task

## Results
Results of our work with link to the code in Jupyter Notebook.
Every task has been trained on three seeds (0,34,42) on n epochs with the default params** (see appendix)   
| **Task** | **Metric** | **GVPGNN** | **GVPTransformer** |
|----------|------------|------------|--------------------|
| **SMP**  |   mae      |            |                    |
| **LBA**  |   RMSE     |            |                    |
| **LEP**  | AUROC      |            |                    |
| **MSP**  |  AUROC     |            |                    |
| **RES**  | Accuracy   |            |                    |

| **Task** | **Metric** | **GVPTransformer** | **GVPTransformer + ProteinBERT** |
|----------|------------|--------------------|----------------------------------|
| **RES**  |  Accuracy  |                    |                                  |

## Conclusion
Conclusion of our work.

## Individual Student's Contribution
Description of what each student's contribution to the project was.

## References

## Appendix
Batch size and Hyper params used --> see notion

