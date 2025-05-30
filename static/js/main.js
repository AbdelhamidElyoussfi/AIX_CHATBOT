/**
 * AIX Systems Assistant - Frontend Optimizations
 * 
 * This file contains optimizations for the frontend to improve performance.
 */

// Cache for storing message content
const messageCache = new Map();
const MAX_CACHE_SIZE = 100;

// Lazy loading for chat history
let isLoadingHistory = false;
let historyPageSize = 10;
let currentHistoryPage = 1;

// Performance metrics
const perfMetrics = {
    responseTime: [],
    renderTime: []
};

/**
 * Add a message to the cache
 * @param {string} id - Message ID
 * @param {object} message - Message object
 */
function cacheMessage(id, message) {
    // Remove oldest entry if cache is full
    if (messageCache.size >= MAX_CACHE_SIZE) {
        const oldestKey = messageCache.keys().next().value;
        messageCache.delete(oldestKey);
    }
    
    messageCache.set(id, message);
}

/**
 * Get a message from the cache
 * @param {string} id - Message ID
 * @returns {object|null} - Message object or null if not found
 */
function getCachedMessage(id) {
    return messageCache.has(id) ? messageCache.get(id) : null;
}

/**
 * Optimize images by lazy loading them
 */
function optimizeImages() {
    const images = document.querySelectorAll('img[data-src]');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                observer.unobserve(img);
            }
        });
    });
    
    images.forEach(img => observer.observe(img));
}

/**
 * Optimize chat rendering by using document fragments
 * @param {Array} messages - Array of message objects
 * @param {HTMLElement} container - Container element
 */
function renderMessagesOptimized(messages, container) {
    const startTime = performance.now();
    
    // Create document fragment for better performance
    const fragment = document.createDocumentFragment();
    
    messages.forEach(message => {
        const messageId = message.id || Date.now().toString();
        
        // Check cache first
        const cachedElement = getCachedMessage(messageId);
        if (cachedElement) {
            fragment.appendChild(cachedElement.cloneNode(true));
            return;
        }
        
        // Create message element
        const messageEl = document.createElement('div');
        messageEl.className = message.isUser ? 'message user-message' : 'message bot-message';
        messageEl.setAttribute('data-message-id', messageId);
        
        // Create message content (simplified for example)
        messageEl.innerHTML = `
            <div class="message-avatar">${message.isUser ? 'U' : 'AIX'}</div>
            <div class="message-bubble">
                <div class="message-content">${message.content}</div>
            </div>
        `;
        
        // Cache the element
        cacheMessage(messageId, messageEl.cloneNode(true));
        
        // Add to fragment
        fragment.appendChild(messageEl);
    });
    
    // Append all messages at once
    container.appendChild(fragment);
    
    // Record performance
    const endTime = performance.now();
    perfMetrics.renderTime.push(endTime - startTime);
    
    // Log performance metrics occasionally
    if (perfMetrics.renderTime.length % 10 === 0) {
        const avgRenderTime = perfMetrics.renderTime.reduce((a, b) => a + b, 0) / perfMetrics.renderTime.length;
        console.log(`Average render time: ${avgRenderTime.toFixed(2)}ms`);
    }
}

/**
 * Load chat history with pagination for better performance
 * @param {string} sessionId - Session ID
 * @param {HTMLElement} container - Container element
 * @param {boolean} reset - Whether to reset the history
 */
async function loadChatHistoryOptimized(sessionId, container, reset = false) {
    if (isLoadingHistory) return;
    
    isLoadingHistory = true;
    
    if (reset) {
        currentHistoryPage = 1;
        container.innerHTML = '';
    }
    
    try {
        const response = await fetch(`/api/history?session=${sessionId}&page=${currentHistoryPage}&limit=${historyPageSize}`);
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            renderMessagesOptimized(data.history, container);
            currentHistoryPage++;
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    } finally {
        isLoadingHistory = false;
    }
}

/**
 * Optimize the send message function with debouncing
 * @param {Function} sendFn - Original send function
 * @param {number} delay - Debounce delay in ms
 * @returns {Function} - Debounced function
 */
function optimizeSendMessage(sendFn, delay = 300) {
    let timeout;
    
    return function(...args) {
        clearTimeout(timeout);
        
        return new Promise((resolve) => {
            timeout = setTimeout(() => {
                const startTime = performance.now();
                
                // Call original function and resolve with its result
                resolve(sendFn.apply(this, args).then(result => {
                    const endTime = performance.now();
                    perfMetrics.responseTime.push(endTime - startTime);
                    return result;
                }));
            }, delay);
        });
    };
}

/**
 * Optimize scrolling performance
 * @param {HTMLElement} container - Scrollable container
 */
function optimizeScrolling(container) {
    let scrollTimeout;
    let isScrolling = false;
    
    container.addEventListener('scroll', () => {
        // Add a class to reduce animations during scrolling
        if (!isScrolling) {
            document.body.classList.add('is-scrolling');
            isScrolling = true;
        }
        
        // Clear the timeout
        clearTimeout(scrollTimeout);
        
        // Set a timeout to remove the class after scrolling stops
        scrollTimeout = setTimeout(() => {
            document.body.classList.remove('is-scrolling');
            isScrolling = false;
        }, 100);
    });
}

/**
 * Initialize optimizations when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', () => {
    // Optimize images
    optimizeImages();
    
    // Optimize scrolling for chat container
    const chatContainer = document.getElementById('chat-messages');
    if (chatContainer) {
        optimizeScrolling(chatContainer);
    }
    
    // Add CSS for scrolling optimization
    const style = document.createElement('style');
    style.textContent = `
        .is-scrolling .message-bubble {
            transition: none !important;
        }
        .is-scrolling .typing-dots .dot {
            animation-play-state: paused !important;
        }
    `;
    document.head.appendChild(style);
    
    console.log('Frontend optimizations initialized');
}); 