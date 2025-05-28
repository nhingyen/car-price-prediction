document.addEventListener('DOMContentLoaded', () => {
    const hamburger: HTMLElement | null = document.getElementById('hamburger');
    const mobileMenu: HTMLElement | null = document.getElementById('mobile-menu');

    if (hamburger && mobileMenu) {
        hamburger.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }
});