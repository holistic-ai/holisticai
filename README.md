<h1 align="center">
<img src="docs/holistic_ai.png" width="100">
<br>Holistic AI
</h1>

The Holistic AI library is an open-source tool to assess and improve the trustworthiness of AI systems.  

Currently, the library offers a set of techniques to easily measure and mitigate Bias across numerous tasks. In the future, it will be extended to include tools for Efficacy, Robustness, Privacy and Explainability as well. This will allow a holistic assessment of AI systems.  

- Documentation: https://holistic-ai.readthedocs.io/en/latest/
- Tutorials: https://github.com/holistic-ai/holisticai/tree/main/tutorials
- Source code: https://github.com/holistic-ai/holisticai/tree/main
- Holistic Ai website: https://holisticai.com


# Installation:
For metrics, you can use the default installation:

```bash
pip install holisticai
pip install holisticai[bias] # For bias visualization and mitigation
```

## Troubleshooting
on **macOS** could be necessary some packages before install holisticai library:
```bash
brew install cbc pkg-config
python -m pip install cylp
brew install cmake
```

## Contributing

We welcome contributions to the Holistic AI library. If you are interested in contributing, please refer to our [contributing guide](CONTRIBUTING.md).
