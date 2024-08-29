==============
Robustness
==============

.. contents:: Table of Contents
   :local:
   :depth: 1

Robustness in machine learning refers to the ability of a model to maintain stable and reliable predictive performance despite variations and challenges in the input data. This concept is crucial for ensuring that models remain effective in real-world scenarios, where the data encountered during deployment often differs from the data seen during training. Variations can include input feature changes, data distribution shifts due to environmental factors, and even deliberate adversarial attacks. A robust machine learning model can resist these disturbances and continue to perform within acceptable limits, thus demonstrating resilience and adaptability in the face of uncertainties. This characteristic is essential for deploying machine learning models in critical applications, where consistent and reliable performance is essential.

On the other hand,  assessing a model's robustness presents a significant challenge since it involves evaluating its performance under natural variations in data and its resilience against these adversarial manipulations. Robust assessment methodologies often include adversarial testing, where models are exposed to crafted adversarial examples to measure their ability to maintain performance. By rigorously evaluating how models respond to expected and adversarial input data changes, we can better understand their robustness and enhance their defenses, leading to more reliable and secure machine learning systems in practical applications.

Adversarial attacks 
~~~~~~~~~~~~~~~~~~~

Adversarial attacks are deliberate strategies used to expose the vulnerabilities of machine learning (ML) models by crafting inputs that deceive the model into making incorrect predictions. These attacks are broadly categorized based on the attacker's knowledge of the target model and their specific objectives. Two primary categories are white-box and black-box attacks. In white-box attacks, the attacker has full access to the model's architecture, parameters, and gradients, allowing them to craft precise adversarial examples that maximize prediction errors. On the other hand, Black-box attacks operate with limited knowledge, relying on input-output queries to infer the model's weaknesses and generate adversarial examples. Additionally, adversarial attacks can be either targeted or non-targeted. Targeted attacks aim to force the model to output a specific, incorrect label, while non-targeted attacks seek to mislead the model into any incorrect prediction. The aim of these attacks is to test and potentially exploit the robustness of ML models, revealing weaknesses that could be critical in high-stakes applications like security or autonomous systems. Understanding these categories and aims is essential for developing more resilient and robust models.
  

White-box Adversarial Attacks
----------------------------

White-box adversarial attacks are sophisticated forms of attack where the attacker has complete access to the target machine learning model's internal details, including its architecture, parameters, and gradients. This extensive knowledge enables attackers to craft highly effective adversarial examples by directly manipulating inputs based on the model's behavior. Typically, white-box attacks utilize gradient-based methods, which involve computing gradients of the model's loss function concerning the input and perturbing the input in the direction that maximizes the loss. This approach ensures that the adversarial example significantly impacts the model's predictions, or applying perturbations iteratively or using random restarts to overcome local minima. The ability to precisely manipulate the input data makes white-box attacks particularly dangerous, as they can exploit detailed knowledge of the model to achieve high levels of attack success.

Black-box Adversarial Attacks
----------------------------

Unlike white-box attacks, where attackers have full knowledge of the model's architecture and parameters, black-box attackers can only interact with the model through input-output queries, operating with limited access to the target model. These approaches enable attackers to infer the model's vulnerabilities by observing how it responds to various inputs. For instance, Zeroth-Order Optimization (ZOO) exploits query responses to estimate gradients, allowing attackers to generate adversarial examples without directly accessing the model's internal mechanics. Despite the challenges posed by the lack of detailed knowledge, black-box attacks can still be highly effective, mainly due to transferability, where adversarial examples crafted on one model can successfully deceive another similar model. This ability highlights the importance of developing robust defenses against adversarial attacks, as black-box strategies mimic real-world scenarios where attackers may not have insider knowledge of the target system.

==============

Attackers and metrics
~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    attackers/hopskipjump
    attackers/zoo
    metrics/index