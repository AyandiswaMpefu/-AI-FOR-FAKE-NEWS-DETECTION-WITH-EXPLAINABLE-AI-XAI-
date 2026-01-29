Abstract

The purpose of this project, Fake News detection is to design and develop an intelligent system capable of accurately distinguishing between factual and fabricated news. The system aims to enhance media literacy, counter misinformation, and restore public trust in digital information ecosystems using explainable and generative AI techniques.

The project employs a hybrid methodology the integrates Natural Language Processing (NLP), deep learning and explainable AI to analyze news articles. The system collects news data, processes it, and trains models such as Random Forest, LSTM, Linear Regression and Naïve bayes to detect fake news. To make the results more transparent, SHAP (an explainable AI tool) is used to show which words or features influenced the model’s decisions.

The system accurately distinguishes between genuine and false news articles. It also provides clear explanations that describe the possible social and emotional impacts of misinformation. Early testing shows major improvements in both detection accuracy and user confidence when compared to older, less transparent AI models.

This system can be used by news organizations, fact-checking groups, and schools to automatically check the credibility of online news. Its explainable AI feature helps people understand how decisions are made, making it an effective tool to fight misinformation and encourage more thoughtful news consumption.

The strength of this project lies in combining AI that both detects and explains fake news. Unlike traditional models that simply give results, this system shows the reasoning behind its predictions, building trust in AI technology and promoting responsible, transparent use in the digital world.


Introduction
The rise of digital media and social networks has revolutionized information sharing but also fuelled the spread of misinformation and fake news, which can distort public opinion and cause social harm. Detecting fake news is challenging due to the massive volume of online content and evolving deception techniques (Broda E et al, 2024). Recent advances in Explainable AI (XAI) provide effective solutions, models like Random Forest and LSTM can analyze and classify news content, while XAI enhances transparency by explaining AI-driven decisions (Dhiman et al., 2024). Combining these technologies enables more accurate and trustworthy fake news detection systems.
Despite advancements in AI-driven content analysis, most fake news detection systems still lack transparency and interpretability (Akter et al., 2025). Many existing models operate as “black boxes,” offering little insight into how decisions are made. This lack of explainability reduces user trust and limits the adoption of such systems.  The continuous evolution of misinformation strategies requires adaptive and intelligent detection methods. Therefore, there is a need for Explainable AI-based framework that can accurately detect provide a clear explanation for its classifications.
Objective: 
•	Employ explainable AI techniques to improve user understanding and transparency.
•	Implement NLP models for precise fake news detection.
•	Evaluate the performance of the proposed models

This study is significant as it contributes to combating misinformation in the digital era by utilizing XAI. By integrating transparency and interpretability into fake news detection, the outcomes of this research could benefit journalists, social media platforms, policymakers, and the public by providing a reliable tool for identifying and explaining misinformation, supports global efforts toward creating a safer and more truthful digital information ecosystem.

4. Literature Review

Literature Review
Theoretical Framework
The theoretical foundation of this study lies in the intersection of artificial intelligence (AI), machine learning (ML), and explainable deep learning theories, emphasizing the need to balance predictive accuracy with model interpretability. Traditional machine learning models, including Logistic Regression, Naïve Bayes, Random Forest, and One-Class Support Vector Machines (O-SVM), are rooted in statistical learning theory, prioritizing transparency, simplicity, and generalization (Hussain et al., 2025). These models provide clear decision boundaries and probabilistic reasoning, making them valuable for tasks requiring interpretability and traceable decision-making processes (Hussain et al., 2025).
In contrast, deep learning models, particularly Long Short-Term Memory (LSTM) networks, are grounded in connectionist learning theories that enable the capture of sequential and contextual dependencies within data (Hashmi et al., 2023). LSTMs have been instrumental in text-based applications such as fake news detection because of their ability to process long-term dependencies (Hashmi et al., 2023). However, while they often surpass traditional models in classification performance, their complex internal representations create a “black-box” problem, limiting human understanding of their decisions (Wang et al., 2024a).
This study’s theoretical stance therefore emphasizes explainable AI (XAI), which seeks to make model predictions more transparent and interpretable without compromising accuracy (Hashmi et al., 2023). By comparing both traditional and deep learning approaches under the same explainability framework, the study aligns with the theoretical goal of developing AI systems that are both powerful and trustworthy (Wang et al., 2024b).
Review of Previous Studies
The widespread circulation of misinformation and fake news across digital platforms has created an urgent need for intelligent, explainable, and trustworthy detection systems. Artificial intelligence has played a central role in this effort, with recent advances in deep learning and large language models (LLMs) transforming how fake news is analysed and detected. Prior studies can be broadly grouped into three key categories:
(A) Explainable Fake News Detection using LLMs,
(B) Deep Learning and Hybrid Models, and
(C) Systematic Reviews of the Fake News Detection Landscape.
Explainable Fake News Detection Using Large Language Models (LLMs)
Recent work highlights the importance of integrating explainability into fake news detection systems. Traditional deep learning approaches, while achieving high accuracy, often lack interpretability, providing little insight into why a particular news article is labelled as true or false (Wang et al., 2024a). To address this, researchers have started using LLMs with prompt-based reasoning and self-reflection mechanisms that can explain predictions in human-readable language (Wang et al., 2024b).
For instance, LLM-GAN: Construct Generative Adversarial Network Through Large Language Models for Explainable Fake News Detection (Wang et al., 2024b) combines the reasoning capabilities of LLMs with a generative adversarial framework. Here, the LLM alternates between acting as a Generator that fabricates fake news and a Detector that identifies deception. Through Inter-Adversary Prompting and Self-Reflection Prompting, the model refines its reasoning and generates logical explanations for its classifications.
Similarly, Explainable Fake News Detection via Defense Among Competing Wisdom (Wang et al., 2024a) introduces the L-Defense approach, which models argumentation between conflicting evidence sources. It divides information into two sets supporting “true” (E⁺) and “false” (E⁻) claims and uses LLMs (e.g., ChatGPT or LLaMA2) to perform abductive reasoning on both sides. The model then compares these justifications based on informativeness and soundness, selecting the strongest argument as the final explanation (Wang et al., 2024a). These studies mark a shift toward detection systems that not only classify misinformation but also rationalize their reasoning.
Deep Learning and Hybrid Models
Another stream of research explores deep learning and hybrid architectures that integrate textual, visual, and contextual information for comprehensive fake news detection. For example, Fake News Detection Based on BERT Multi-domain and Multi-modal Fusion Network (Yu et al., 2025) combines BERT for textual features with VGG-19 for image features, constructing a Union Matrix of Similarities and a Union Matrix of Attention to model interactions between modalities. Additionally, an Event Domain Adaptive Network aligns multimodal features across different events, addressing cross-domain variability. The model achieved superior results on Weibo and Twitter datasets, outperforming other multimodal approaches (Yu et al., 2025).
On the other hand, Advancing Fake News Detection: Hybrid Deep Learning with FastText and Explainable AI (Hashmi et al., 2023) proposes a hybrid model that integrates Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks with FastText embeddings. The CNN extracts local spatial features, while the LSTM captures temporal dependencies. Using explainability techniques such as Local Interpretable Model-Agnostic Explanations (LIME) and Latent Dirichlet Allocation (LDA), the model provides interpretable insights into its decision-making process. This combination achieved an F1-score of 0.99 on the WELFake dataset, illustrating how XAI tools can complement deep learning performance with interpretability (Hashmi et al., 2023).
Systematic Reviews of the Field
Comprehensive reviews of fake news detection research provide valuable insight into the field’s evolution. For example, Fake News Detection Landscape: Datasets, Data Modalities, AI Approaches, Their Challenges, and Future Perspectives (Hussain et al., 2025) synthesizes findings from over 310 studies published between 2017 and 2025. It documents a clear methodological shift from traditional machine learning techniques (e.g., TF-IDF, Logistic Regression, Naïve Bayes) to transformer-based architectures like BERT and RoBERTa. However, it also highlights persistent challenges, including language imbalance with most datasets in English and inconsistent application of explainable AI methods. These findings suggest a growing need for multilingual, multimodal, and interpretable detection models (Hussain et al., 2025).
Gap in the Literature
While prior research demonstrates considerable success in both traditional and deep learning approaches to fake news detection, several gaps remain. Most comparative studies focus primarily on model performance metrics such as accuracy, precision, and recall without adequately addressing model explainability or interpretability (Hashmi et al., 2023; Wang et al., 2024a). Additionally, many studies apply explainability frameworks exclusively to LLMs or deep models, overlooking traditional machine learning approaches that are more computationally efficient and accessible (Hussain et al., 2025).
This study addresses these gaps by conducting a comparative analysis of multiple machine learning and deep learning algorithms namely LSTM, Random Forest, Logistic Regression, Naïve Bayes, and One-Class SVM to identify the most effective fake news detection model. Importantly, explainability techniques are then applied only to the best-performing model, thereby enhancing interpretability without compromising computational efficiency. Through this approach, the study contributes to the growing body of research seeking to balance performance, transparency, and trustworthiness in AI-driven misinformation detection systems.


Methodology

1)	Research Design: Describe the overall Project
This study adopts an experimental research design centered on the development and evaluation of an explainable AI-based fake news detection system. The research integrates Natural Language Processing (NLP), supervised and unsupervised learning models, and explainability mechanisms to enhance interpretability. A quantitative evaluation framework was used to assess model performance, interpretability, and robustness.

2)	Data Collection Methods: Explain how data were collected.
The dataset comprises both real and fake news articles sourced from reputable and non-reputable platforms. True news articles were obtained by crawling Reuters.com, while fake news articles were collected from sources identified by Politifact and Wikipedia as unreliable. The dataset primarily includes political and world news topics, ensuring diversity and realism in content representation.


3)	Data Analysis Techniques: Describe the methods used to analyze the data.
1.	Model Development
Five core models were implemented:
1)	Supervised Models: Logistic Regression, Naive Bayes, Random Forest, and Long Short-Term Memory (LSTM).
2)	Unsupervised Models: One-Class Support Vector Machine (OCSVM) and Isolation Forest, both designed to detect anomalies in text embeddings.
 
Each model was trained and tuned based on its inherent strengths:
1)	Random Forest and Logistic Regression: Trained on TF-IDF vectors to capture frequency-based linguistic cues.
2)	LSTM: Employed word embeddings to learn temporal dependencies within text sequences.
3)	OCSVM and Isolation Forest: Operated on high-dimensional embeddings to identify outliers indicative of fake news patterns.
2. Explainable AI Integration
To address the challenge of transparency, an explainability layer was implemented using occlusion-based interpretability techniques. This method systematically perturbed portions of input text to identify words or phrases most influential in model predictions. The explanations were visualized locally within the web-based prototype, allowing users to view how textual evidence contributed to classification outcomes.
3. Model Evaluation
Model performance was quantitatively assessed using:
     1)    Accuracy, Precision, Recall F1-Score: To measure classification   effectiveness.
     2)    Confusion Matrices: To visualize correct and incorrect predictions.
     3)    Explainability Metrics: To evaluate the quality and interpretability of model explanations.
  4. System Implementation
The models were deployed within a Django-based web framework. The system architecture included:
      1)   A URL input interface for submitting news links.
      2)   A text extraction module utilizing Trafilatura and BeautifulSoup.
      3)   A backend inference API for classification.
      4)   A local explanation engine displaying interpretability visualizations.
5. Validation and Limitations
Testing revealed that while the models achieved exceptional performance on the dataset, potential overfitting and label leakage may exist. These issues highlight the necessity for:
     1)     Cross-validation.
     2)     Evaluation on temporally distinct and external datasets.
     3)     Probability calibration and adversarial robustness testing.
