// Smart Laptop Buying Assistant — Frontend Interactions
document.addEventListener('DOMContentLoaded', () => {
    // Form submission loading state
    const form = document.querySelector('form[action*="predict"]');
    if (form) {
        form.addEventListener('submit', (e) => {
            const submitBtn = form.querySelector('.analyze-submit-btn');
            if (submitBtn) {
                const btnText = submitBtn.querySelector('.btn-text');
                const spinner = submitBtn.querySelector('.spinner-icon');
                if (btnText && spinner) {
                    btnText.textContent = 'Analyzing Hardware & Estimating Price...';
                    spinner.style.display = 'inline-block';
                    submitBtn.style.opacity = '0.9';
                    submitBtn.style.pointerEvents = 'none';
                }
            }
        });
    }

    // Smooth scroll for hero CTA
    const scrollLinks = document.querySelectorAll('a[href^="#"]');
    scrollLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const targetId = link.getAttribute('href');
            if (targetId && targetId !== '#') {
                const targetElem = document.querySelector(targetId);
                if (targetElem) {
                    e.preventDefault();
                    targetElem.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
});
