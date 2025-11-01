---
layout: page
title: "About me"
permalink: /about/
---

<style>
.hero-section {
  display: flex;
  align-items: center;
  gap: 2rem;
  margin-bottom: 3rem;
  padding: 2rem;
  background: linear-gradient(135deg, #f0f7ff 0%, #e8f4f8 100%);
  border-radius: 16px;
  border-left: 4px solid #3851cf;
}

.hero-image {
  flex-shrink: 0;
}

.hero-image img {
  width: 180px;
  height: 180px;
  border-radius: 50%;
  object-fit: cover;
  border: 4px solid white;
  box-shadow: 0 8px 24px rgba(0,0,0,0.1);
}

.hero-content h1 {
  margin-top: 0;
  color: #1a1a2e;
  font-size: 2rem;
}

.hero-tagline {
  font-size: 1.1rem;
  color: #555;
  line-height: 1.6;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.stat-card {
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  border: 2px solid #e0e0e0;
  text-align: center;
  transition: transform 0.3s, box-shadow 0.3s;
}

.stat-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.1);
  border-color: #3851cf;
}

.stat-number {
  font-size: 2rem;
  font-weight: 700;
  color: #3851cf;
  display: block;
}

.stat-label {
  color: #666;
  font-size: 0.9rem;
  margin-top: 0.5rem;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 3rem;
  margin-bottom: 1.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 3px solid #3851cf;
}

.section-header h2 {
  margin: 0;
  font-size: 1.75rem;
}

.skills-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.skill-category {
  background: #f9f9f9;
  padding: 1.5rem;
  border-radius: 12px;
  border-left: 4px solid #3851cf;
}

.skill-category h3 {
  margin-top: 0;
  font-size: 1.1rem;
  color: #3851cf;
}

.skill-category p {
  margin: 0;
  line-height: 1.8;
  color: #444;
}

.milestone-card {
  background: white;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  border-radius: 12px;
  border-left: 4px solid #3851cf;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.milestone-card h3 {
  margin-top: 0;
  color: #3851cf;
}

.books-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.book-item {
  background: #f9f9f9;
  padding: 1rem 1.25rem;
  border-radius: 8px;
  border-left: 3px solid #3851cf;
  font-size: 0.95rem;
}

@media (max-width: 768px) {
  .hero-section {
    flex-direction: column;
    text-align: center;
  }
  
  .hero-content h1 {
    font-size: 1.5rem;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>

<img src="../images/profile.webp" alt="Ayushi Jain" style="width:150px; float:right; margin:0 0 1rem 1rem; border-radius: 50%;">

Hi, I'm **Ayushi Jain**, a Software Engineer at **Microsoft** with a curious mind and a passion for building highly scalable, intelligent systems that learn, reason, and collaborate adhering to principles of high reliability and security.

<div class="section-header">
  <h2>🔗 Connect With Me</h2>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.5rem 0;">
  <a href="https://github.com/ayushi2019031" target="_blank" style="display: inline-flex; align-items: center; padding: 0.75rem 1.25rem; background: #24292e; color: white; text-decoration: none; border-radius: 8px; font-weight: 500; transition: transform 0.2s, box-shadow 0.2s;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
    <svg style="width: 20px; height: 20px; margin-right: 0.5rem; fill: currentColor;" viewBox="0 0 16 16">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
    </svg>
    GitHub
  </a>
  
  <a href="https://linkedin.com/in/ayushi31" target="_blank" style="display: inline-flex; align-items: center; padding: 0.75rem 1.25rem; background: #0077b5; color: white; text-decoration: none; border-radius: 8px; font-weight: 500; transition: transform 0.2s, box-shadow 0.2s;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
    <svg style="width: 20px; height: 20px; margin-right: 0.5rem; fill: currentColor;" viewBox="0 0 24 24">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
    LinkedIn
  </a>
  
  <a href="mailto:atallakshaya@gmail.com" style="display: inline-flex; align-items: center; padding: 0.75rem 1.25rem; background: #ea4335; color: white; text-decoration: none; border-radius: 8px; font-weight: 500; transition: transform 0.2s, box-shadow 0.2s;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
    <svg style="width: 20px; height: 20px; margin-right: 0.5rem; fill: currentColor;" viewBox="0 0 24 24">
      <path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/>
    </svg>
    Email
  </a>
  
  <a href="/public-files/Ayushi_Jain_Resume.pdf" target="_blank" style="display: inline-flex; align-items: center; padding: 0.75rem 1.25rem; background: #3851cf; color: white; text-decoration: none; border-radius: 8px; font-weight: 500; transition: transform 0.2s, box-shadow 0.2s;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
    <svg style="width: 20px; height: 20px; margin-right: 0.5rem; fill: currentColor;" viewBox="0 0 24 24">
      <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
    </svg>
    Resume
  </a>
</div>

---

At **Azure**, my work spans across **A2A and MCP protocol integrations**, large-scale cloud automation across 20+ regions, and secure B2B authentication flows. Along the way, I continue to strengthen my **core software engineering skills** in distributed systems, backend architecture, and DevOps.

I began my journey at Microsoft as an intern, where I optimized machine learning libraries on single machine, and built PoCs for distributed ML using **PySpark** and **Databricks**, and since then, my curiosity for scalable and intelligent systems has only grown stronger.

Before joining Microsoft, I earned my **B.Tech. in Computer Science Engineering** from **IIIT Delhi** (CGPA: 8.34), where I developed a deep interest in **distributed systems**, **machine learning**, and **deep learning**. I continue to explore these areas through courses like **Stanford's XCS234 (Reinforcement Learning)** and **XCS236 (Deep Generative Models)**, **Azure certifications**, **Kaggle competitions** and **independent exploration**.

Outside of work, I enjoy reading fiction and non-fiction, and experimenting with side projects that combine **AI, systems design, and creativity**.  
This blog is my space to document what I learn and build, not just as a record of progress but as a way to make **AI and engineering more approachable** for others.

This site is powered by curiosity, continuous learning, and the belief that **knowledge grows best when shared**.

---

## 🎯 Professional Milestones

- **SWE @ Microsoft (L59 → L60)** — *2023 - Present*   
Working in Azure Specialized org building Agentic AI systems

- **Microsoft Azure Certifications** — 2024
  - **[Azure Administrator Associate (AZ-104)](https://learn.microsoft.com/en-us/credentials/certifications/azure-administrator/?practice-assessment-type=certification)** — Skilled in managing Azure identities, governance, storage, and compute resources across distributed cloud environments.
  - **[Azure Developer Associate (AZ-201)](https://learn.microsoft.com/en-us/credentials/certifications/azure-developer/?practice-assessment-type=certification)** — Experienced in designing scalable, secure, and resilient Azure architectures for enterprise workloads.
  - **[Azure AI Engineer Associate (AI-102)](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-engineer/)** — Proficient in building, deploying, and integrating AI services, including NLP, vision, and conversational AI models on Azure.
  - **[Cosmos DB Developer Specialty (DP-420)](https://learn.microsoft.com/en-us/credentials/certifications/azure-cosmos-db-developer-specialty/)** — Specialized in building globally distributed, high-performance applications using Azure Cosmos DB and multi-region data replication.  


- **Stanford University (Advanced AI Courses)** — *Stanford Online, 2024*
  - **[CS234: Reinforcement Learning](https://digitalcredential.stanford.edu/check/3EA90D2274EE4BB33FFFBB2C93091C16880FBA58DF11119933D828B32AF3868DVzFTOEJtVitJN2ROWlZqS3RXeit6V0FVeDltUWFselVQUTI1UjNWNHllSDdrbTQv)** (*Winter 2024*) — Covered foundational and advanced RL techniques including policy gradients, Q-learning, and bandit algorithms, with a focus on sequential decision-making under uncertainty.
  - **[CS236: Deep Generative Models](https://digitalcredential.stanford.edu/check/A94BC56E7F6F1330A926CCD1833998FCEE640CCC8DF13D041DF7A1AC42E69AEDLzk3aWt0b0tUNy9HdmJ3aDE4ejNZS0IyUHF5ZHVVQW5DRzRtTjI0cjR0R0lHcXIz)** (*Summer 2024*) — Explored generative modeling techniques such as VAEs, GANs, Normalizing Flows, and Diffusion Models, emphasizing both theory and implementation.

- **SWE Intern @Microsoft** — *Microsoft, Summer 2022*
  - **Optimization of Microsoft's internal Machine learning library**: written using deep learning libraries and algorithms in python based on paper written by MSFT Research leading to reduction in memory usage by 75% on 100+ GB datasets using time profiling, vectorization and reduction in logging.
  - Built a **distributed ML PoC using PySpark and Databricks** demonstrating scalable pre-processing and training on large-scale datasets.
  - Secured a return offer from Microsoft 😊

- **Dean's Award for Excellence in Teaching Assistantship** — *IIIT Delhi, 2022*
  - Received appreciation for mentoring 400+ students of junior batches in IIIT Delhi in DSA
  - As a part of TAship, prepared assignments, took tutorials and office hours for students.

- **Machine Learning Engineer** — *Wunderman Thompson Salmon, 2021*
  - Implemented a recommendation system to predict customers' next baskets based on the paper using PyTorch.

- **Research Publication** — *ACM Hypertext Conference, 2022*
  - Published research paper at [ACM HT'22](https://dl.acm.org/doi/10.1145/3511095.3536368)

---

## 🛠️ Technical Skills

**Languages & Frameworks**  
Python • C# • .NET • JavaScript • Java • KQL • C++ • ReactJS 

**AI & Machine Learning**  
PyTorch • TensorFlow • Scikit-learn • LangChain • Semantic Kernel • Azure AI Foundry • OpenAI APIs • RAG Systems • Agent Architectures • Reinforcement Learning

**Cloud & Infrastructure**  
Azure (Cosmos DB, Azure Functions, App Services, Authentication, Azure ML) • Docker • Kubernetes • CI/CD (GitHub Actions, Azure DevOps)

**Tools & Platforms**  
Git • VS Code • Jupyter • 

---

## 🎓 Academic Background

**IIIT Delhi** | B.Tech. in Computer Science Engineering | 2019 - 2023  
*Courses: Machine learning, Natural Language Processing, Collaborative Filtering, Distributed Systems,  Theory of Computation, Advanced Data Structures and Algorithms, Computer Networks, Operating Systems*  
CGPA: **8.34/10**

**Class 12 (CBSE)** | DAV Public School, Shreshtha Vihar, Delhi  | 2019  
*Subjects: Physics, Chemistry, Mathematics, English, Physical Education*  
Percentage: **95.4%**

**NTSE Scholar (National Talent Scholar Examination - Class 10th)** | 2017  
*About the exam: It is conducted by the National Council of Educational Research and Training (NCERT) under the Government of India. It is designed for students in Class 10 (or equivalent) in India. The exam is taken in two stages - at state and national level. There are three papers at every stage: Mat (Mental Aptitude Test), English Language Test, SAT (Scholastic Aptitude Test) - covering topics ranging from logical reasoning, physical and social sciences.*   
Among **top 750 students in India** in class 10th to clear the exam.  

**Class 10 (CBSE)** | Cambridge School, Indirapuram | 2014  
CGPA: **10/10**

## Test Scores

**JEE Advanced** | Joint Engineering Entrance (Advanced) Exam | 2019  
*Top 0.3 percentile amongst 100k applicants*  
AIR: **11404**  

**JEE Mains** | Joint Engineering Entrance Exam | 2019    
*Top 0.27 percentile amongst a million applicants*  
AIR: **3366**

---

## 📚 Books I Have Recently Read

- The Silent Patient, *Alex Michaelides*  
- All The Light We Cannot See, *Anthony Doerr*  
- The Story Of My Life, *Helen Keller*  
- More Days At Morisaki Bookshop, *Satoshi Yagasawa*  
- Days At The Morisaki Bookshop, *Satoshi Yagasawa*  
- The Murder of Roger Ackroyd, *Agatha Christie*   
- And Then There Were None, *Agatha Christie*  
- How To Kill Your Family, *Bella Mackie*


---

*Thanks for stopping by — and if you’re building something cool in AI, I’d love to hear from you! Please feel free to reach out at [atallakshaya@gmail.com](atallakshaya@gmail.com)*  
*This is my personal website, and all thoughts posted here are only mine and not linked to the workplace I am at.*

