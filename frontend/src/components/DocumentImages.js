import React, { useState, useEffect, useCallback } from 'react';
import { getDocumentImages, getDocumentImageUrl } from '../services/api';

/**
 * Component to display images extracted from a document
 */
const DocumentImages = ({ documentId, isVisible = true }) => {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(null);

  /**
   * Fetch images for the document
   */
  const fetchImages = async () => {
    if (!documentId || !isVisible) return;

    setLoading(true);
    setError(null);

    try {
      const response = await getDocumentImages(documentId);
      setImages(response.images || []);
      
      if (response.images && response.images.length > 0) {
        console.log(`Loaded ${response.images.length} images for document ${documentId}`);
      }
    } catch (err) {
      console.error('Error fetching document images:', err);
      setError(err.message || 'Failed to load document images');
    } finally {
      setLoading(false);
    }
  };

  // Fetch images when component mounts or documentId changes
  useEffect(() => {
    fetchImages();
  }, [documentId, isVisible]);

  /**
   * Handle image click to show in modal
   */

  const handleImageClick = (image, index) => {
    setSelectedImage(image);
    setSelectedIndex(index);
  };

  // Download image utility
  const handleDownload = (image) => {
    const url = getDocumentImageUrl(documentId, image.filename);
    const link = document.createElement('a');
    link.href = url;
    link.download = image.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Modal navigation
  const showPrev = useCallback(() => {
    if (selectedIndex > 0) {
      setSelectedImage(images[selectedIndex - 1]);
      setSelectedIndex(selectedIndex - 1);
    }
  }, [selectedIndex, images]);

  const showNext = useCallback(() => {
    if (selectedIndex < images.length - 1) {
      setSelectedImage(images[selectedIndex + 1]);
      setSelectedIndex(selectedIndex + 1);
    }
  }, [selectedIndex, images]);

  // Keyboard navigation for modal
  useEffect(() => {
    if (!selectedImage) return;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') closeModal();
      if (e.key === 'ArrowLeft') showPrev();
      if (e.key === 'ArrowRight') showNext();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedImage, showPrev, showNext]);

  /**
   * Close image modal
   */
  const closeModal = () => {
    setSelectedImage(null);
  };

  // Don't render if not visible
  if (!isVisible) {
    return null;
  }

  // Loading state
  if (loading) {
    return (
      <div className="document-images-section">
        <h3 className="text-lg font-semibold mb-3 text-gray-800">Document Images</h3>
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Loading images...</span>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="document-images-section">
        <h3 className="text-lg font-semibold mb-3 text-gray-800">Document Images</h3>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700">Error loading images: {error}</p>
          <button
            onClick={fetchImages}
            className="mt-2 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // No images
  if (images.length === 0) {
    return (
      <div className="document-images-section">
        <h3 className="text-lg font-semibold mb-3 text-gray-800">Document Images</h3>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <p className="text-gray-600">No images were extracted from this document.</p>
        </div>
      </div>
    );
  }

  // Render images
  return (
    <div className="document-images-section">
      <h3 className="text-lg font-semibold mb-3 text-gray-800">
        Document Images ({images.length})
      </h3>
      {/* Image grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 mb-6 p-4 bg-gray-50 rounded-xl border border-gray-100">
        {images.map((image, index) => (
          <div key={index} className="relative flex items-center justify-center group">
            <div
              className="bg-white rounded-2xl overflow-hidden cursor-pointer border border-gray-200 shadow-sm hover:shadow-xl hover:border-blue-400 transition-all w-44 h-28 md:w-60 md:h-32 lg:w-72 lg:h-36 flex items-center justify-center"
              onClick={() => handleImageClick(image, index)}
            >
              <img
                src={getDocumentImageUrl(documentId, image.filename)}
                alt={`Document image ${index + 1}`}
                className="w-full h-full object-cover group-hover:scale-105 group-hover:opacity-90 transition-transform transition-opacity duration-200"
                onError={(e) => {
                  e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik04NiA3Mkw4NiAxMjhMMTE0IDEyOEwxMTQgNzJIODZaIiBmaWxsPSIjOUI5QjlCIi8+CjxwYXRoIGQ9Ik04NiA3Mkw4NiAxMjhMMTE0IDEyOEwxMTQgNzJIODZaIiBmaWxsPSIjOUI5QjlCIi8+CjwvZz4KPC9zdmc+';
                }}
              />
              {/* Download button in grid */}
              <button
                onClick={e => { e.stopPropagation(); handleDownload(image); }}
                className="absolute top-2 right-2 bg-white bg-opacity-90 rounded-full p-1.5 shadow-md hover:bg-blue-500 hover:text-white transition-all border border-gray-200 hover:border-blue-500"
                title="Download image"
                tabIndex={-1}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V4" /></svg>
              </button>
              {/* Image info overlay */}
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-black/0 text-white text-xs p-2 rounded-b-xl opacity-0 group-hover:opacity-100 transition-opacity flex justify-between items-center">
                <span className="truncate max-w-[70%] font-medium drop-shadow">{image.filename}</span>
                <span className="ml-2 drop-shadow">{Math.round(image.size / 1024)} KB</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Image modal with navigation and download */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="relative max-w-4xl max-h-full flex flex-col items-center">
            <button
              onClick={closeModal}
              className="absolute -top-10 right-0 text-white hover:text-gray-300 text-2xl"
              title="Close (Esc)"
            >
              ✕
            </button>
            {/* Prev button */}
            {selectedIndex > 0 && (
              <button
                onClick={showPrev}
                className="absolute left-0 top-1/2 -translate-y-1/2 bg-black bg-opacity-40 hover:bg-opacity-70 text-white rounded-full p-2 text-2xl"
                style={{ left: '-3rem' }}
                title="Previous (←)"
              >
                &#8592;
              </button>
            )}
            {/* Next button */}
            {selectedIndex < images.length - 1 && (
              <button
                onClick={showNext}
                className="absolute right-0 top-1/2 -translate-y-1/2 bg-black bg-opacity-40 hover:bg-opacity-70 text-white rounded-full p-2 text-2xl"
                style={{ right: '-3rem' }}
                title="Next (→)"
              >
                &#8594;
              </button>
            )}
            <img
              src={getDocumentImageUrl(documentId, selectedImage.filename)}
              alt={selectedImage.filename}
              className="max-w-full max-h-[70vh] object-contain rounded-lg shadow-lg border border-white"
            />
            <div className="w-full flex flex-col md:flex-row md:items-center md:justify-between bg-black bg-opacity-80 text-white p-4 rounded-b-lg mt-2">
              <div>
                <h4 className="font-semibold text-base md:text-lg">{selectedImage.filename}</h4>
                <p className="text-xs md:text-sm text-gray-300">Size: {Math.round(selectedImage.size / 1024)} KB</p>
              </div>
              <button
                onClick={() => handleDownload(selectedImage)}
                className="mt-2 md:mt-0 bg-white bg-opacity-90 hover:bg-opacity-100 text-gray-800 px-4 py-2 rounded shadow flex items-center gap-2"
                title="Download image"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V4" /></svg>
                Download
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentImages;
