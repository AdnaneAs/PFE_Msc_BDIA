import React, { useState, useEffect } from 'react';
import { getDocumentImages, getDocumentImageUrl } from '../services/api';

/**
 * Component to display images extracted from a document
 */
const DocumentImages = ({ documentId, isVisible = true }) => {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

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
  const handleImageClick = (image) => {
    setSelectedImage(image);
  };

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
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-4">
        {images.map((image, index) => (
          <div key={index} className="relative group">
            <div
              className="aspect-square bg-gray-100 rounded-lg overflow-hidden cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => handleImageClick(image)}
            >
              <img
                src={getDocumentImageUrl(documentId, image.filename)}
                alt={`Document image ${index + 1}`}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                onError={(e) => {
                  e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik04NiA3Mkw4NiAxMjhMMTE0IDEyOEwxMTQgNzJIODZaIiBmaWxsPSIjOUI5QjlCIi8+CjxwYXRoIGQ9Ik04NiA3Mkw4NiAxMjhMMTE0IDEyOEwxMTQgNzJIODZaIiBmaWxsPSIjOUI5QjlCIi8+CjwvZz4KPC9zdmc+';
                }}
              />
            </div>
            
            {/* Image info overlay */}
            <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white text-xs p-2 rounded-b-lg opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="truncate">{image.filename}</p>
              <p>{Math.round(image.size / 1024)} KB</p>
            </div>
          </div>
        ))}
      </div>

      {/* Image modal */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="relative max-w-4xl max-h-full">
            <button
              onClick={closeModal}
              className="absolute -top-10 right-0 text-white hover:text-gray-300 text-2xl"
            >
              ✕
            </button>
            
            <img
              src={getDocumentImageUrl(documentId, selectedImage.filename)}
              alt={selectedImage.filename}
              className="max-w-full max-h-full object-contain rounded-lg"
            />
            
            <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white p-4 rounded-b-lg">
              <h4 className="font-semibold">{selectedImage.filename}</h4>
              <p className="text-sm text-gray-300">
                Size: {Math.round(selectedImage.size / 1024)} KB
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentImages;
