/**
 * Performance monitoring utility for configuration loading
 */

class ConfigPerformanceMonitor {
  constructor() {
    this.metrics = {
      cacheHits: 0,
      cacheMisses: 0,
      loadTimes: [],
      errors: 0
    };
    this.startTimes = new Map();
  }

  /**
   * Start timing a configuration load operation
   * @param {string} operation Operation identifier
   */
  startTiming(operation) {
    this.startTimes.set(operation, performance.now());
  }
  /**
   * End timing and record the result
   * @param {string} operation Operation identifier
   * @param {boolean} fromCache Whether data was loaded from cache
   */
  endTiming(operation, fromCache = false) {
    const startTime = this.startTimes.get(operation);
    if (startTime) {
      const duration = performance.now() - startTime;
      
      // Ensure loadTimes array exists
      if (!Array.isArray(this.metrics.loadTimes)) {
        this.metrics.loadTimes = [];
      }
      
      this.metrics.loadTimes.push(duration);
      
      if (fromCache) {
        this.metrics.cacheHits++;
      } else {
        this.metrics.cacheMisses++;
      }
      
      this.startTimes.delete(operation);
      
      console.log(`🚀 Config ${operation}: ${duration.toFixed(1)}ms ${fromCache ? '(cached)' : '(fresh)'}`);
    }
  }

  /**
   * Record an error
   */
  recordError() {
    this.metrics.errors++;
  }
  /**
   * Get performance statistics
   * @returns {Object} Performance metrics
   */
  getStats() {
    // Ensure loadTimes array exists
    if (!Array.isArray(this.metrics.loadTimes)) {
      this.metrics.loadTimes = [];
    }
    
    const totalLoads = this.metrics.cacheHits + this.metrics.cacheMisses;
    const cacheHitRate = totalLoads > 0 ? (this.metrics.cacheHits / totalLoads * 100).toFixed(1) : 0;
    const avgLoadTime = this.metrics.loadTimes.length > 0 
      ? (this.metrics.loadTimes.reduce((a, b) => a + b, 0) / this.metrics.loadTimes.length).toFixed(1)
      : 0;

    return {
      ...this.metrics,
      totalLoads,
      cacheHitRate: `${cacheHitRate}%`,
      avgLoadTime: `${avgLoadTime}ms`,
      recentLoadTimes: this.metrics.loadTimes.slice(-5).map(t => `${t.toFixed(1)}ms`)
    };
  }

  /**
   * Log current performance stats to console
   */
  logStats() {
    const stats = this.getStats();
    console.group('📊 Configuration Performance Stats');
    console.log('Cache Hit Rate:', stats.cacheHitRate);
    console.log('Average Load Time:', stats.avgLoadTime);
    console.log('Total Loads:', stats.totalLoads);
    console.log('Errors:', stats.errors);
    console.log('Recent Load Times:', stats.recentLoadTimes);
    console.groupEnd();
  }

  /**
   * Reset all metrics
   */
  reset() {
    this.metrics = {
      cacheHits: 0,
      cacheMisses: 0,
      loadTimes: [],
      errors: 0
    };
    this.startTimes.clear();
  }
}

// Export singleton instance
export default new ConfigPerformanceMonitor();
