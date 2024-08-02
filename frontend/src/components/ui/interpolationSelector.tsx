import { useState } from 'react';
import { ComboBox, ComboBoxOption } from '@shadcn/ui';

const InterpolationSelector = () => {
  const [selectedCategory, setSelectedCategory] = useState('');
  const [interpolationOptions, setInterpolationOptions] = useState([]);

  const handleCategoryChange = (value) => {
    setSelectedCategory(value);
    if (value === 'Spatial') {
      setInterpolationOptions([
        'Nearest Neighbor',
        'Bilinear',
        'Bicubic',
      ]);
    } else if (value === 'Temporal') {
      setInterpolationOptions([
        'Linear',
        'Cubic Spline',
        'Polynomial',
      ]);
    } else {
      setInterpolationOptions([]);
    }
  };

  return (
    <div>
      <ComboBox
        label="Select Category"
        onChange={handleCategoryChange}
      >
        <ComboBoxOption value="Spatial">Spatial</ComboBoxOption>
        <ComboBoxOption value="Temporal">Temporal</ComboBoxOption>
      </ComboBox>

      {selectedCategory && (
        <ComboBox label={`Select ${selectedCategory} Interpolation Technique`}>
          {interpolationOptions.map((option, index) => (
            <ComboBoxOption key={index} value={option}>
              {option}
            </ComboBoxOption>
          ))}
        </ComboBox>
      )}
    </div>
  );
};

export default InterpolationSelector;
