"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

type Option = {
  value: string;
  label: string;
};

interface ComboboxPopoverProps {
  options: Option[];
  value: Option | null;
  onChange: (value: Option) => void;
  placeholder?: string;
}

export default function ComboboxPopover({
  options = [],
  value,
  onChange,
  placeholder,
}: ComboboxPopoverProps) {
  const [open, setOpen] = React.useState(false);
  const [selectedOption, setSelectedOption] = React.useState<Option | null>(value);

  React.useEffect(() => {
    setSelectedOption(value);
  }, [value]);

  return (
    <div className="flex items-center space-x-4">
      <p className="text-sm text-muted-foreground">
        <b>METHODS</b>
      </p>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" className="w-[550px] justify-start">
            {selectedOption ? <>{selectedOption.label}</> : <>+ Select</>}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="p-0" side="right" align="start">
          <Command>
            <CommandInput placeholder={placeholder} />
            <CommandList>
              <CommandEmpty>No results found.</CommandEmpty>
              <CommandGroup>
                {options.map((option) => (
                  <CommandItem
                    key={option.value}
                    value={option.value}
                    onSelect={(value) => {
                      const selected = options.find((opt) => opt.value === value) || null;
                      setSelectedOption(selected);
                      onChange(selected as Option);
                      setOpen(false);
                    }}
                  >
                    {option.label}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
