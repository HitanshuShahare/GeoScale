// components/UploadForm.tsx
"use client"
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const UploadForm = () => {
  const [imageSrc, setImageSrc] = useState<string>('');

  useEffect(() => {
    const fetchImage = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/generate');
        setImageSrc(response.data.imageUrl);
      } catch (error) {
        console.error('Error fetching image:', error);
      }
    };

    fetchImage();
  }, []);

  const handleDownload = () => {
    if (imageSrc) {
      const link = document.createElement('a');
      link.href = imageSrc;
      link.download = 'enhanced_image.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="p-4">
      {imageSrc && (
        <div className="mt-4">
          <img src={imageSrc} alt="Generated" className="transform transition-transform duration-300"/>
          <div className="mt-2 space-x-2">
            <button onClick={handleDownload} className="bg-gray-500 text-white py-2 px-4 rounded">Download</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadForm ;
