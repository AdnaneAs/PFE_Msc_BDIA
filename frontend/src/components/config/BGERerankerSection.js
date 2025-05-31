import React from 'react';

const BGERerankerSection = ({
  rerankerConfig,
  useReranking,
  rerankerModel,
  onRerankingToggle,
  onRerankerModelChange,
  disabled
}) => {
  if (!rerankerConfig) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg">
        <div className="p-6">
          <div className="text-sm text-gray-500">Loading BGE reranker configuration...</div>
        </div>
      </div>
    );
  }

  const { available_models, benchmark_summary } = rerankerConfig;

  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">BGE Reranking Configuration</h3>
        <p className="text-sm text-gray-600 mt-1">
          Enhanced document relevance scoring with BGE reranker models
        </p>
      </div>

      <div className="p-6 space-y-6">
        {/* Performance Banner */}
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="w-5 h-5 text-green-600 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h4 className="text-sm font-medium text-green-800">Proven Performance Improvements</h4>
              <div className="mt-1 text-sm text-green-700">
                Academic benchmark results ({benchmark_summary.test_dataset}):
              </div>
              <ul className="mt-2 text-xs text-green-600 space-y-1">
                <li>• <strong>+{benchmark_summary.base_model_improvements.map_improvement_percent}%</strong> MAP (Mean Average Precision)</li>
                <li>• <strong>+{benchmark_summary.base_model_improvements.precision_at_5_improvement_percent}%</strong> Precision@5</li>
                <li>• <strong>+{benchmark_summary.base_model_improvements.ndcg_at_5_improvement_percent}%</strong> NDCG@5</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Enable/Disable Toggle */}
        <div>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700">
                Enable BGE Reranking
              </label>
              <p className="text-sm text-gray-600 mt-1">
                Improve document relevance scoring with state-of-the-art BGE reranker models
              </p>
            </div>
            <button
              onClick={() => !disabled && onRerankingToggle(!useReranking)}
              disabled={disabled}
              className={`
                relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                ${useReranking ? 'bg-green-600' : 'bg-gray-200'}
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              <span
                className={`
                  inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                  ${useReranking ? 'translate-x-6' : 'translate-x-1'}
                `}
              />
            </button>
          </div>
          {useReranking && (
            <div className="mt-3 p-3 bg-green-50 rounded-md">
              <div className="text-sm text-green-800">
                ✓ BGE reranking is enabled
              </div>
              <div className="text-xs text-green-600 mt-1">
                Documents will be reranked for improved relevance (+{benchmark_summary.base_model_improvements.map_improvement_percent}% MAP improvement)
              </div>
            </div>
          )}
        </div>

        {/* Model Selection */}
        {useReranking && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">
              BGE Reranker Model
            </label>
            <div className="space-y-3">
              {Object.values(available_models).map((model) => (
                <button
                  key={model.name}
                  onClick={() => !disabled && onRerankerModelChange(model.name)}
                  disabled={disabled}
                  className={`
                    w-full p-4 rounded-lg border-2 text-left transition-all duration-200
                    ${rerankerModel === model.name
                      ? 'border-green-500 bg-green-50 text-green-900'
                      : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                    }
                    ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                  `}
                >
                  <div className="flex items-start">
                    <div className={`
                      w-4 h-4 rounded-full border-2 mr-3 mt-0.5
                      ${rerankerModel === model.name
                        ? 'border-green-500 bg-green-500'
                        : 'border-gray-300'
                      }
                    `}>
                      {rerankerModel === model.name && (
                        <div className="w-full h-full rounded-full bg-white scale-50"></div>
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-sm">{model.display_name}</div>
                      <div className="text-xs text-gray-600 mt-1">{model.description}</div>
                      <div className="flex items-center mt-2 space-x-4 text-xs text-gray-500">
                        <span>Size: {model.model_size}</span>
                        <span>Speed: {model.speed}</span>
                        <span>Quality: {model.quality}</span>
                      </div>
                      {model.benchmark_results && typeof model.benchmark_results.map_improvement === 'number' && (
                        <div className="mt-2 text-xs text-green-600">
                          Benchmark: +{model.benchmark_results.map_improvement}% MAP improvement
                        </div>
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Performance Settings Info */}
        {useReranking && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h5 className="text-sm font-medium text-blue-800 mb-2">How BGE Reranking Works</h5>
            <ul className="text-xs text-blue-700 space-y-1">
              <li>• Initial retrieval finds {rerankerConfig.performance_settings?.initial_retrieval_multiplier || 3}x more documents than requested</li>
              <li>• BGE reranker scores each document for relevance to your query</li>
              <li>• Top documents are selected based on enhanced relevance scores</li>
              <li>• Results in significantly more accurate and relevant answers</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default BGERerankerSection;
