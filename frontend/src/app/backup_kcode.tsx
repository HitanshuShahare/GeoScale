'use client';

import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import ComboboxPopover from "@/components/ui/combobox-popover";
import { Slider } from "@/components/ui/slider";
import jsPDF from 'jspdf';
import { FaExpand, FaCompress, FaSearchPlus, FaSearchMinus } from 'react-icons/fa';
import dynamic from 'next/dynamic';
import 'leaflet/dist/leaflet.css';

const MapContainer = dynamic(() => import('react-leaflet').then((mod) => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then((mod) => mod.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then((mod) => mod.GeoJSON), { ssr: false });

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
  { value: "rbf", label: "Radial Basis Function" },
  { value: "nearest", label: "Nearest Neighbor" },
  { value: "linear", label: "Linear Interpolation" },
  { value: "natural_neighbor", label: "Natural Neighbor" },
  { value: "thin_plate_spline", label: "Thin Plate Spline" },
  { value: "universal_kriging", label: "Universal Kriging" },
  { value: "cokriging", label: "Co-Kriging" },
  { value: "trend_surface", label: "Trend Surface Analysis" },
];

const temporalTechniques: Option[] = [
  { value: "linear", label: "Linear Interpolation" },
  { value: "spline", label: "Spline Interpolation" },
  { value: "arima", label: "ARIMA" },
  { value: "exponential_smoothing", label: "Exponential Smoothing" },
  { value: "prophet", label: "Prophet" },
  { value: "kalman_filter", label: "Kalman Filter" },
  { value: "advanced_forecast", label: "Advanced Forecast" },
  { value: "gaussian_process", label: "Gaussian Process Regression" },
  { value: "ssa", label: "Singular Spectrum Analysis" },
  { value: "sarima", label: "Seasonal ARIMA (SARIMA)" },
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
  const [choroplethData, setChoroplethData] = useState<any>(null);

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
        setChoroplethData(JSON.parse(data.choroplethData));
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
      pdf.addImage(image, 'PNG', 10, 10, 190, 100);
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
        <div className="z-50 relative">
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
        <div className="z-50 relative">
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

      {image && choroplethData && (
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
            <ChoroplethMap 
              data={choroplethData} 
              zoomLevel={zoomLevel}
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

interface ChoroplethMapProps {
  data: any;
  zoomLevel: number;
}

const ChoroplethMap: React.FC<ChoroplethMapProps> = ({ data, zoomLevel }) => {
  const mapStyle = (feature: any) => {
    return {
      fillColor: getColor(feature.properties.temperature),
      weight: 2,
      opacity: 1,
      color: 'white',
      dashArray: '3',
      fillOpacity: 0.7
    };
  };

  const getColor = (value: number) => {
    return value > 30 ? '#800026' :
           value > 25 ? '#BD0026' :
           value > 20 ? '#E31A1C' :
           value > 15 ? '#FC4E2A' :
           value > 10 ? '#FD8D3C' :
           value > 5  ? '#FEB24C' :
           value > 0  ? '#FED976' :
                        '#FFEDA0';
  };

  if (!data || !data.features || data.features.length === 0) {
    return <div>No data available for the map</div>;
  }

  const bounds = data.features.reduce((acc: any, feature: any) => {
    const coords = feature.geometry.coordinates;
    if (coords && coords.length === 2) {
      if (!acc) {
        return [[coords[1], coords[0]], [coords[1], coords[0]]];
      }
      return [
        [Math.min(acc[0][0], coords[1]), Math.min(acc[0][1], coords[0])],
        [Math.max(acc[1][0], coords[1]), Math.max(acc[1][1], coords[0])]
      ];
    }
    return acc;
  }, null);

  return (
    <MapContainer 
      bounds={bounds}
      style={{ height: '500px', width: '100%' }}
      zoomControl={false}
      zoom={zoomLevel}
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <GeoJSON data={data} style={mapStyle} />
    </MapContainer>
  );
};

export default Home;