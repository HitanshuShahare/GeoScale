'use client';

import React, { useRef, useMemo, useState } from 'react';
import { Button } from "@/components/ui/button";
import ComboboxPopover from "@/components/ui/combobox-popover";
import jsPDF from 'jspdf';
import { FaExpand, FaCompress, FaSearchPlus, FaSearchMinus } from 'react-icons/fa';

interface Option {
  value: string;
  label: string;
}

const interpolationTypes: Option[] = [
  { value: "spatial", label: "Spatial Interpolation" },
  { value: "temporal", label: "Temporal Interpolation" },
];

const spatialTechniques: Option[] = [
  { value: "idw", label: "Inverse Distance Weighting (IDW)" },
  { value: "kriging", label: "Kriging" },
  { value: "natural_neighbor", label: "Natural Neighbor" },
  { value: "spline", label: "Spline" },
  { value: "trend", label: "Trend Surface Analysis" },
];

const temporalTechniques: Option[] = [
  { value: "linear", label: "Linear Interpolation" },
  { value: "polynomial", label: "Polynomial Interpolation" },
  { value: "spline", label: "Spline Interpolation" },
  { value: "kalman", label: "Kalman Filter" },
  { value: "arima", label: "ARIMA Model" },
];

const Home: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [interpolationType, setInterpolationType] = useState<Option | null>(null);
  const [technique, setTechnique] = useState<Option | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isFullScreen, setIsFullScreen] = useState<boolean>(false);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const imageRef = useRef<HTMLImageElement>(null);

  const techniques = useMemo(() => {
    if (!interpolationType) return [];
    return interpolationType?.value === 'spatial' ? spatialTechniques : temporalTechniques;
  }, [interpolationType]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!file || !interpolationType || !technique) {
      setError("Please select all required fields.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('interpolationType', interpolationType.value);
    formData.append('technique', technique.value);

    try {
      const response = await fetch('http://localhost:5000/interpolate', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        setImage(URL.createObjectURL(blob));
      } else {
        console.error('Error:', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleGeostatistical1Submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!file || !interpolationType || !technique) {
      setError("Please select all required fields.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('interpolationType', interpolationType.value);
    formData.append('technique', technique.value);

    try {
      const response = await fetch('http://localhost:5000/generate_geostatistical_image', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        setImage(URL.createObjectURL(blob));
      } else {
        console.error('Error:', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleGeostatisticalSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!file || !interpolationType || !technique) {
      setError("Please select all required fields.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('interpolationType', interpolationType.value);
    formData.append('technique', technique.value);

    try {
      const response = await fetch('http://localhost:5000/downscaled_geostatistical_image', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        setImage(URL.createObjectURL(blob));
      } else {
        console.error('Error:', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleDownloadPDF = (): void => {
    if (image && imageRef.current) {
      const pdf = new jsPDF();
      const imgProps = imageRef.current;
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = imgProps.naturalWidth;
      const imgHeight = imgProps.naturalHeight;
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
      const imgX = (pdfWidth - imgWidth * ratio) / 2;
      const imgY = 30;

      pdf.addImage(image, 'JPEG', imgX, imgY, imgWidth * ratio, imgHeight * ratio);
      pdf.save('interpolation_result.pdf');
    }
  };

  const toggleFullScreen = () => {
    setIsFullScreen(!isFullScreen);
  };

  const handleZoomIn = () => {
    setZoomLevel(prevZoom => Math.min(prevZoom + 0.1, 3));
  };

  const handleZoomOut = () => {
    setZoomLevel(prevZoom => Math.max(prevZoom - 0.1, 0.5));
  };

  return (
    <div className="container">
      <main className="max-w-2xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-center">Interpolation App</h1>

        <form onSubmit={handleSubmit} className="space-y-4 mb-8">
          <div>
            <label htmlFor="file" className="block text-sm font-medium text-gray-700 mb-1">
              CSV File:
            </label>
            <input
              type="file"
              id="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              required
              className="mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Interpolation Type:
            </label>
            <ComboboxPopover 
              options={interpolationTypes}
              value={interpolationType}
              onChange={(value) => {
                setInterpolationType(value);
                setTechnique(null);
              }}
              placeholder="Select interpolation type"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Interpolation Technique:
            </label>
            <ComboboxPopover 
              options={techniques}
              value={technique}
              onChange={setTechnique}
              placeholder="Select interpolation technique"
            />
          </div>
          <Button type="submit" className="w-full" disabled={!file || !interpolationType || !technique}>
            Generate Interpolation Image
          </Button>
        </form>

        <form onSubmit={handleGeostatistical1Submit} className="space-y-4">
          <Button type="submit" className="w-full" disabled={!file || !interpolationType || !technique}>
            Normal Geostatistical Image
          </Button>
        </form>

        <form onSubmit={handleGeostatisticalSubmit} className="space-y-4">
          <Button type="submit" className="w-full" disabled={!file || !interpolationType || !technique}>
            Generate Downscaled Geostatistical Image
          </Button>
        </form>

        {error && (
          <div className="mt-4 text-red-600 text-center">
            {error}
          </div>
        )}

        {image && (
          <div className={`mt-8 ${isFullScreen ? 'fixed inset-0 z-50 bg-white' : ''}`}>
            <div className="flex justify-end space-x-2 mb-2">
              <Button onClick={handleZoomOut}>
                <FaSearchMinus />
              </Button>
              <Button onClick={handleZoomIn}>
                <FaSearchPlus />
              </Button>
              <Button onClick={toggleFullScreen}>
                {isFullScreen ? <FaCompress /> : <FaExpand />}
              </Button>
            </div>
            <div className={`overflow-auto ${isFullScreen ? 'h-[calc(100vh-60px)]' : 'max-h-[600px]'}`}>
              <img 
                ref={imageRef}
                src={image} 
                alt="Interpolation result" 
                className={`w-full ${isFullScreen ? 'h-auto object-contain' : 'rounded-lg shadow-lg'}`}
                style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }}
              />
            </div>
            {!isFullScreen && (
              <Button onClick={handleDownloadPDF} className="w-full mt-4">
                Download as PDF
              </Button>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default Home;