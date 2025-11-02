// Theme toggle functionality
(function() {
  const themeToggle = document.getElementById('theme-toggle');
  const html = document.documentElement;
  
  // Get saved theme from localStorage or default to light
  const savedTheme = localStorage.getItem('theme') || 'light';
  html.setAttribute('data-theme', savedTheme);
  
  // Update toggle button state
  function updateToggleButton(theme) {
    const sunIcon = themeToggle.querySelector('.sun-icon');
    const moonIcon = themeToggle.querySelector('.moon-icon');
    
    if (theme === 'dark') {
      sunIcon.style.display = 'none';
      moonIcon.style.display = 'block';
    } else {
      sunIcon.style.display = 'block';
      moonIcon.style.display = 'none';
    }
  }
  
  // Initialize button state
  updateToggleButton(savedTheme);
  
  // Toggle theme on button click
  themeToggle.addEventListener('click', function() {
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateToggleButton(newTheme);
  });
})();
