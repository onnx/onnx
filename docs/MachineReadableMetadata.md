# ONNX Model metadata

## Abstract

This documentation builds on [the core metadata recommendations](MetadataProps.md) to suggest a workflow that includes *machine-readable* metadata to support rich queries, most specifically tracking provenance, but also supporting documenting associated concerns such as possible bias in training data, Responsible AI and similar.

## Introduction

In what follows we describe a method to enrich and structure metadata (which is to be machine-readable) to aid model provenance tracking, discoverability of models on hubs, annotation of fairness/Ethical AI considerations and other concerns (in an extensible manner).  

## Goals

* Model details, Domain, Datasets, Applicable areas, Fairness, Transparency, Human centered approach, Trusted, Secure, Privacy protections, ..
* Facilitate metadata checks to be implemented in runtime(s) as part of Preprocessing element of ONNX Graph
* Leverage Semantic Web infrastructure (RDF) for defining the structured metadata to keep it extensible, machine readable, Embeddable (compatible with existing ONNX metadata fields). Rely on industry wide initiatives for introducing Controlled vocabularies for Provenance, Explainable and Ethical ML are already being developed (eg [IEEE P7003](https://standards.ieee.org/project/7003.html))
* Enables queries to be performed on model zoo to filter for relevant use cases

Simplify the user interactions with multiple variants hosted in ONNX model zoo and allow the ONNX community to simplify sharing from various industry contributors.

## ONNX Model provenance metadata

Expand on the fields implemented for model hub implementation. The metadata fields hold optional information not required for the basic level of functionality

This should be considered as an initial proposal for metadata fields to capture model provenance &amp; mixed precision representation. We encourage a broader discussion around what metadata should be included in the official model zoo when they get published.

## Example Metadata for model provenance and mixed precision

Initial proposal (noncomprehensive) list of metadata attached &amp; associated with each model that gets published in the zoo:

<table>
  <tr>
    <th>Category</th><th>Name</th><th>Type</th><td>Note</th>
  </tr>
<tr><td>Model details</td><td>Name</td><td>String</td></tr>
<tr><td></td><td>Domain</td><td>String</td><td>Vision, Text, NLP, Graph NN etc.. </td></tr>
<tr><td></td><td>Objective</td><td>String</td><td>Classification, Regression, Forecast, Cluster, Security etc… </td></tr>
<tr><td></td><td>Precisions</td><td>Double</td><td>FP32, FP16, Int8, FP32+Int8, FP16 + Int8, etc… </td></tr>
<tr><td></td><td>Inputs</td><td>Map[String, NodeInfo]</td><td>Allows models within a domain to have the same default API</td></tr>
<tr><td></td><td>Outputs</td><td>Map[String, NodeInfo]</td><td>Allows models within a domain to have the same default API </td></tr>
<tr><td></td><td>Preprocessing</td><td>String</td><td>Instruction for preprocessing, such as input normalization</td></tr>
<tr><td></td><td>Postprocessing</td><td>String</td><td>Instruction for postprocessing, such as SoftMax</td></tr>
<tr><td></td><td>Data Provenance</td><td>Training data</td><td>URL</td></tr>
<tr><td></td><td>Testing data</td><td>URL</td></tr>
<tr><td></td><td>Number of training examples</td><td>Double</td></tr>
<tr><td></td><td>Number of features/dimensionality</td><td>Double</td></tr>
<tr><td></td><td>[Training] Environment</td><td>URL</td></tr>
<tr><td></td><td>Geo groups</td><td>URL</td></tr>
<tr><td></td><td>Trainer Provenance/Trainer algorithm</td><td>URL</td><td>Capture specifics like Single-ensemble-1, etc.. </td></tr>
<!--
 | Hyperparameters | URL | Capture specifics like Learning rate, etc.. |
|
 | Statistics dashboard | URL |
 |
| Metrics | Top1-Accuracy | Double |
 |
|
 | Top5-Accuracy | Double |
 |
|
 | Average Precision | Double |
 |
| Applications | Scope | String |
 |
|
 | Out-of-Scope | String |
 |
| Authors | Name | String | Individual, Company etc.. |
|
 | Origination-Date | String | MM-DD-YYYY |
| Citation | Citations | URL |
 |
| Documentation | Documents | URL |
 |
| Licensing | License | URL |
-->
</table>

## Metadata management mechanics

The ONNX format, and its protobuf realization already has some support for decorating models with metadata. We propose here to build on this support to provide a more standardized management of these fields. Following the guidelines of &quot;Model Cards&quot; [1], we encode metadata relevant to model deployment using the well-known, and widely-adopted, Markdown format. However, the proposal in [1] concentrated on the use of metadata by a human operator. We extend this by augmenting the Markdown with a machine-readable block (taking advantage of the support markdown provides for &quot;frontmatter blocks&quot;) which serializes an RDF fragment incorporating the metadata fields described in the sections above. In this way, annotating a ONNX model with this &quot;enriched metadata&quot; by a model creator would take a form similar to:
<pre>
model = onnxmltools.load_model(&quot;model.onnx&quot;)
meta = model.metadata_props.add()
meta.key = "model_card"
meta.value = &lt;markdown-with-frontmatter string&gt;
onnxmltools.utils.save_model(model, &quot;model_with_metadata.onnx&quot;)
</pre>

The frontmatter we recommend in this proposal is a serialization of an RDF fragment (using, for example, the ontology of FAIRNets [2]. Markdown frontmatter content most usually takes the form of yaml-encoded information, and there are several proposed yaml serializations for RDF (for example [aref](https://gbv.github.io/aREF/aREF.html)). RDF enables for a decentralized metadata model, allowing, for example, for the metadata to consist simply of a stable URL for the ONNX model user to refer to for a more complete description (to allay, to some extent, overhead concerns).

A valuable use case that this design supports is for model repositories (&quot;hubs&quot; or &quot;zoos&quot;) to provide query endpoints (in the manner of [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page)) that would mine their models for their metadata, and provide their users with a powerful tool for discovery and selection of the model that is best suited for their applications and risk profiles (including fairness, interpretability, ecological impact and similar considerations). Some examples of these queries (encoded in [SPARQL](https://www.w3.org/TR/sparql11-query/)) are shown below, with nno as an alias for the FAIRnet namespace:

List models for NLP tasks trained (or suitable) for texts that describe sports events

<pre>
SELECT  ?label ?category  WHERE  {​

?network a nno:NeuralNetwork;​

         rdfs:label ?label;​

         nno:hasIntendedUse &quot;nlp&quot;;​

         doap:category ?category.​

FILTER ( REGEX ( LCASE ( STR (?category)), &quot;sports&quot;))​

}
</pre>

List models, their creator and the number of layers (as an indicator for model size)
<pre>
SELECT  ?label ?creator ?link ( COUNT (?layer))  WHERE  {​

?network a nno:NeuralNetwork;​

         rdfs:label ?label;​

         dc:creator ?creator;​

         nno:hasRepository ?link;​

         nno:hasModel ?model.​

?model nno:hasLayer ?layer.​

}
</pre>

List models including a metric for their training carbon footprint (note that this information is not part of the FAIRNet ontology, so it serves as an illustration of how to extend the framework)

<pre>
SELECT  ?label ?link ?emission  WHERE  {​

?network a nno:NeuralNetwork;​

         rdfs:label ?label;​

         nno:hasRepository ?link;​

  temp:co2emissions ?emission.​

}  ORDER BY ?emis
</pre>

## Related work

As mentioned above, support for free-form metadata attachment is already supported in the ONNX format, and several tools that export to ONNX already make use of this support to decorate the models they export with basic information like the name of the tool and the date of creation. Similarly, ONNX model visualizers like [Netron](https://netron.app/) are also able to decode and display metadata in a text box, without making any assumptions on the format.

There are other efforts that strive to furnish ONNX models with more structured metadata. Most notable for the purposes of this discussion is the work to document provenance in the [Tribuo](https://tribuo.org/) framework [3]. This work seems to have a relatively tight binding with the Java platform and the [olcut library](https://github.com/oracle/olcut) from Oracle. In this way, it provides a very attractive and smooth workflow for model developers to include annotations. We strongly believe that such compelling developer experience is key for model creators to adopt whatever metadata functionality is provided is on offer. In this way, it seems to us that a solution that combines the benefits of the Tribuo provenance subsystem and the RDF-based metadata proposed here would hit the balance of extensibility and ease of use (for provenance and beyond) that would be attractive to model creators.

## References

[1] Margaret Mitchell et al: &quot;Model cards for model reporting&quot;, in Proceedings of the conference on fairness, accountability, and transparency, 2019

[2] Anna Nguyen et al: &quot;Making neural networks FAIR&quot;, in Proceedings of the Iberoamerican Knowledge Graphs and Semantic Web Conference, 2020

[3] Adam Pocock: &quot;Tribuo: Machine Learning with Provenance in Java&quot;, [https://arXiv.org/abs/2110.03022](https://arXiv.org/abs/2110.03022), 2021
