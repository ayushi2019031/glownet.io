# WARP

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Glownet** is a Jekyll-powered technical blog focused on machine learning, AI systems, and deep learning. The site features in-depth articles on topics like GANs, agentic AI systems, and technical explorations in ML. It's built as a personal blog by Ayushi Jain, a Microsoft Software Engineer specializing in Azure and AI systems.

## Development Commands

### Local Development
```bash
# Install dependencies
bundle install

# Start development server with live reload
bundle exec jekyll serve

# Start development server with drafts included
bundle exec jekyll serve --drafts

# Build for production
bundle exec jekyll build

# Build with future posts (posts with future dates)
bundle exec jekyll build --future
```

### Content Management
```bash
# Create a new blog post
touch _posts/YYYY-MM-DD-title-slug.md

# Check bundle status
bundle check

# Update dependencies
bundle update
```

## Architecture & Structure

### Jekyll Configuration
- **Theme**: Uses Minima theme for clean, readable blog layout
- **Future Posts**: Enabled in `_config.yml` to show posts with future dates
- **Plugins**: Jekyll-feed for RSS generation
- **Domain**: Hosted at glownet.io with custom CNAME

### Content Structure
```
/
├── _posts/                 # Blog articles (markdown with frontmatter)
├── _config.yml            # Jekyll configuration
├── images/                # Static assets and images
├── about.markdown         # About page
├── index.md              # Homepage
└── Gemfile               # Ruby dependencies
```

### Blog Post Format
All blog posts follow this naming convention and frontmatter structure:
```yaml
---
layout: page
title: "Article Title"
permalink: /url-slug/
---
```

### Content Focus Areas
1. **Machine Learning Architectures**: Deep technical dives into models like GANs
2. **AI Systems**: Explanations of modern AI agent architectures and patterns  
3. **Technical Tutorials**: Step-by-step guides with interactive elements
4. **Personal Learning**: Documentation of research and professional development

### Special Features
- **Interactive Elements**: Uses custom CSS for flip cards and interactive components
- **Technical Diagrams**: Includes custom webp images for architecture diagrams
- **Detailed Tables**: Comparison tables for different ML techniques and architectures
- **Code Examples**: Syntax-highlighted code blocks with proper formatting

## Development Notes

### Content Creation Workflow
1. Create new post in `_posts/` with proper date prefix
2. Add required frontmatter (layout, title, permalink)
3. Test locally with `bundle exec jekyll serve`
4. Images should be placed in `images/` directory and referenced relatively
5. Use future dates in post filenames if content is scheduled

### Theme Customization
- Built on Minima theme with custom CSS for interactive elements
- Profile image displayed on homepage via inline styling
- Custom styling for flip cards and technical content presentation

### Site Deployment
- Hosted on GitHub Pages (indicated by CNAME file)
- Static site generation via Jekyll build process
- Custom domain: glownet.io

This is a content-focused Jekyll blog optimized for technical writing, particularly in the AI/ML domain. The structure prioritizes readability and includes interactive elements to enhance learning.
