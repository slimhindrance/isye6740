document.addEventListener('DOMContentLoaded', () => {
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === 1) {  // Check if it's an element node
                    const overlayText = node.querySelector('.image-overlay-text');
                    if (overlayText && overlayText.textContent.includes('Chainlit')) {
                        overlayText.textContent = 'TA Assist';
                    }
                }
            });
        });
    });

    // Observe changes to the body (or a more specific element if needed)
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});