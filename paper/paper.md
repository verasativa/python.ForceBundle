---
title: 'python.ForceBundle: a Python library to reduce cluttering on network visualizations'
tags:
  - networks
  - visualizations
  - Python
authors:
  - name: Vera Sativa
    orcid: 0000-0001-9138-7375
    affiliation: 1
affiliations:
 - name: Fellow student, Bibloteca Gabriela Mistral, Ñuñoa, RM, Chile
   index: 1
date: 5 September 2019
bibliography: paper.bib
---

# Background

Graphs depicted as node-link diagrams are widely used to show relationships between entities. However, node-link diagrams comprised of a large number of nodes and edges often suffer from visual clutter. The use of edgebundling remedies this and reveals high-level edge patterns.

Force-directed Edge Bungling [@holten2009force] uses a self-organizing approach to bundling in which edges are modeled as flexible springs that can attract each other. The resulting bundled graphs show significant clutter reduction and clearly visible high-level edge patterns. Curvature variation is furthermore minimized, resulting in smooth bundles that are easy to follow.


![Comparative of raw lines and same lines processed with FEB](../doc_assets/trips-comparative.png)

# Summary
``python.ForceBundle``
 - Whats is for?
 - Who is for?
 - Features:
    - Hyper-parameters
    - Weights

## Acknowledgements
The author was funded as part of Fondecyt grant xxxx

# References