'use client';

import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import ComboboxPopover from "@/components/ui/combobox-popover";
import { Slider } from "@/components/ui/slider";
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
  { value: "kriging", label: "Ordinary Kriging" },
  { value: "universal_kriging", label: "Universal Kriging" },
  { value: "rbf", label: "Radial Basis Function" },
  { value: "gpr", label: "Gaussian Process Regression" },
  { value: "random_forest", label: "Random Forest Regression" },
  { value: "natural_neighbor", label: "Natural Neighbor Interpolation" },
];

const temporalTechniques: Option[] = [
  { value: "linear", label: "Linear Interpolation" },
  { value: "spline", label: "Spline Interpolation" },
  { value: "arima", label: "ARIMA" },
  { value: "sarima", label: "SARIMA" },
  { value: "exponential_smoothing", label: "Exponential Smoothing" },
  { value: "prophet", label: "Prophet" },
];

const Home: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [interpolationType, setInterpolationType] = useState<Option | null>(null);
  const [technique, setTechnique] = useState<Option | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isFullScreen, setIsFullScreen] = useState<boolean>(false);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [timeStep, setTimeStep] = useState<number[]>([0]);
  const [maxTimeStep, setMaxTimeStep] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    if (interpolationType?.value === 'temporal' && maxTimeStep > 0) {
      setTimeStep([Math.min(timeStep[0], maxTimeStep)]);
    }
  }, [interpolationType, maxTimeStep]);

  const handleTimeStepChange = (value: number[]) => {
    setTimeStep(value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);
    if (!file || !interpolationType || !technique) {
      setError("Please select all required fields.");
      setIsLoading(false);
      return;
    }
    const formData = new FormData();
    formData.append('file', file);
    formData.append('interpolationType', interpolationType.value);
    formData.append('technique', technique.value);
    formData.append('timeStep', timeStep[0].toString());
    try {
      const response = await fetch('http://localhost:5000/generate_climate_map', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        setImage(data.image);
        setMaxTimeStep(data.maxTimeStep || 0);
      } else {
        const errorData = await response.json();
        setError(errorData.error || "An error occurred while generating the climate map.");
      }
    } catch (error) {
      setError("An error occurred while connecting to the server.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadPDF = (): void => {
    if (image) {
      const pdf = new jsPDF();
      const imgProps = new Image();
      imgProps.src = image;
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = imgProps.width;
      const imgHeight = imgProps.height;
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
      const imgX = (pdfWidth - imgWidth * ratio) / 2;
      const imgY = 30;

      pdf.addImage(image, 'PNG', imgX, imgY, imgWidth * ratio, imgHeight * ratio);
      pdf.save('climate_map.pdf');
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
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8 text-center">Climate Map Generator</h1>

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
            options={interpolationType?.value === 'spatial' ? spatialTechniques : temporalTechniques}
            value={technique}
            onChange={setTechnique}
            placeholder="Select interpolation technique"
          />
        </div>
        
        {interpolationType?.value === 'temporal' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Time Step:
            </label>
            <Slider
              value={timeStep}
              onValueChange={handleTimeStepChange}
              max={maxTimeStep}
              step={1}
              min={0}
              className="w-full"
            />
            <div className="mt-1 text-sm text-gray-500">
              Current Time Step: {timeStep[0]}
            </div>
          </div>
        )}
        
        <Button type="submit" className="w-full" disabled={!file || !interpolationType || !technique || isLoading}>
          {isLoading ? 'Generating...' : 'Generate Climate Map'}
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
              src={image} 
              alt="Climate map" 
              className={`w-full ${isFullScreen ? 'h-auto object-contain' : 'rounded-lg shadow-lg'}`}
              style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }}
            />
          </div>
          {!isFullScreen && (
            <div className="mt-4">
              <Button onClick={handleDownloadPDF} className="w-full">
                Download as PDF
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Home;