# Mini-project

![](./images/1200px-DHRS7B_homology_model.png)

## Introduction
In recent years, the field of machine learning has been increasingly applied to the analysis and understanding of 3D structures of large biomolecules. This emerging area holds great promise for advancing our knowledge in fields such as biochemistry, structural biology, and drug discovery (such as finding a COVID vaccin). Despite the growing interest, there is still a lack of a comprehensive network architecture that effectively combines the geometric and relational aspects inherent in this domain.

To bridge this gap, the paper titled "Learning from Protein Structure with Geometric Vector Perceptrons" by Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael J.L. Townshend, and Ron O. Dror from Stanford University introduces a novel approach that addresses the limitations of existing architectures. The authors propose the use of geometric vector perceptrons (GVP's), which extend the functionality of standard dense layers to operate on collections of Euclidean vectors. By integrating these layers into graph neural networks, the proposed approach enables simultaneous geometric and relational reasoning on efficient representations of macromolecules.

The main objective of this study is to demonstrate the effectiveness of GVP's in tackling two fundamental challenges in learning from protein structure: model quality assessment and computational protein design. Through comprehensive experiments and evaluations, the authors compare their approach with existing classes of architectures, including state-of-the-art convolutional neural networks and graph neural networks. The results showcase their state-of-the-art performance of their proposed method in both problem domains, highlighting its potential to advance the field of learning from protein structure.

In this review, we will provide an analysis of the key components of the paper, discussing its strengths, weaknesses, and potential implications. Additionally, we will present our group's response to the paper, highlighting our novel contribution and its relevance to the research presented. Finally, we will delve into the results obtained by our work, linking them to the code available in the accompanying Jupyter notebook. By the end of this blogpost, readers will gain a comprehensive understanding of the significance of the paper's findings and our contributions to the field.

Before proceeding further, it is essential to review related work in the area of learning from protein structure, setting the stage for the novel approach proposed by Jing et al.

# Methods 

## Weaknesses/Strengths/Potential
Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response.

## Novel Contribution
Description of Novel contributions of our work.

## Results
Results of our work with link to the code in Jupyter Notebook.

| **Task** | **Metric** | **GVPGNN** | **GVPTransformer** |
|----------|------------|------------|--------------------|
| **SMP**  |            |            |                    |
| **LBA**  |            |            |                    |
| **LEP**  |            |            |                    |
| **MSP**  |            |            |                    |
| **RES**  |            |            |                    |

| **Task** | **Metric** | **GVPTransformer** | **GVPTransformer + ProteinBERT** |
|----------|------------|--------------------|----------------------------------|
| **RES**  |            |                    |                                  |

## Conclusion
Conclusion of our work.

## Individual Student's Contribution
Description of what each student's contribution to the project was.
