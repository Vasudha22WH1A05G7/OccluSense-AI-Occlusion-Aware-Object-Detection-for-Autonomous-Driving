/**
 * OccluSense AI - Main JavaScript
 * Common utility functions and initializations
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add active class to current nav item
    const currentPath = window.location.pathname;
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });

    // Global loading indicator
    window.showLoading = function(element, text = 'Processing...') {
        const originalContent = element.innerHTML;
        element.disabled = true;
        element.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            ${text}
        `;
        return originalContent;
    };

    window.hideLoading = function(element, originalContent) {
        element.disabled = false;
        element.innerHTML = originalContent;
    };

    // Format numbers
    window.formatNumber = function(num, decimals = 2) {
        if (num === null || num === undefined) return '--';
        return parseFloat(num).toFixed(decimals);
    };

    // Format percentage
    window.formatPercent = function(num, decimals = 0) {
        if (num === null || num === undefined) return '--';
        return (parseFloat(num) * 100).toFixed(decimals) + '%';
    };

    // Copy to clipboard
    window.copyToClipboard = function(text) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy:', err);
        });
    };

    // Show toast notification
    window.showToast = function(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container') || createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => toast.remove());
    };

    function createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
        return container;
    }

    // Debounce function
    window.debounce = function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    };

    // Throttle function
    window.throttle = function(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    };

    // Parse JSON safely
    window.safeJSON = function(str, fallback = null) {
        try {
            return JSON.parse(str);
        } catch (e) {
            return fallback;
        }
    };

    // Chart.js default options
    if (typeof Chart !== 'undefined') {
        Chart.defaults.color = '#9ca3af';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
        Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
    }

    console.log('🚗 OccluSense AI initialized');
});

/**
 * API Helper functions
 */
const API = {
    async detect(formData) {
        const response = await fetch('/api/detect/', {
            method: 'POST',
            body: formData
        });
        return response.json();
    },

    async detectRealtime(frameData, options = {}) {
        const response = await fetch('/api/detect/realtime/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                frame: frameData,
                nms_method: options.nmsMethod || 'standard',
                confidence: options.confidence || 0.25,
                iou_threshold: options.iouThreshold || 0.45,
                sigma: options.sigma || 0.5
            })
        });
        return response.json();
    },

    async getResults(sessionId) {
        const response = await fetch(`/api/results/${sessionId}/`);
        return response.json();
    },

    async getMetrics(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const response = await fetch(`/api/metrics/?${queryString}`);
        return response.json();
    },

    async compareNMS(formData) {
        const response = await fetch('/api/compare-nms/', {
            method: 'POST',
            body: formData
        });
        return response.json();
    }
};

/**
 * Color utilities for detection classes
 */
const ClassColors = {
    person: '#6366f1',
    bicycle: '#22c55e',
    car: '#f59e0b',
    motorcycle: '#ef4444',
    bus: '#06b6d4',
    truck: '#8b5cf6',
    'traffic light': '#ec4899',
    'stop sign': '#14b8a6',
    default: '#9ca3af'
};

function getClassColor(className) {
    return ClassColors[className.toLowerCase()] || ClassColors.default;
}

/**
 * Drawing utilities for canvas
 */
const CanvasUtils = {
    drawBox(ctx, box, label, color, confidence) {
        const [x1, y1, x2, y2] = box;
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw label background
        const text = `${label} ${(confidence * 100).toFixed(0)}%`;
        ctx.font = '14px Inter, sans-serif';
        const textWidth = ctx.measureText(text).width;
        
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - 20, textWidth + 10, 20);
        
        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x1 + 5, y1 - 5);
    },

    drawDetections(ctx, detections) {
        detections.forEach(det => {
            const color = getClassColor(det.class_name);
            this.drawBox(ctx, det.bbox, det.class_name, color, det.confidence);
        });
    }
};

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API, ClassColors, CanvasUtils };
}
