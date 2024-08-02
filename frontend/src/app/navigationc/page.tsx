"use client";
import axios from 'axios';
import { ChangeEvent, useRef, ElementRef, useEffect, useState } from "react";
import React from 'react';
import { cn } from "@/lib/utils";
import { ChevronLeftIcon } from "@radix-ui/react-icons";
import { usePathname } from "next/navigation";
import { useMediaQuery } from "usehooks-ts";
import { MenuIcon } from "lucide-react";
import { NavigationMenuDemo } from "../navigationMenu";
import { Button } from "@/components/ui/button";
import { ComboboxPopover } from "../popOverDemo/page";
import { ComboboxPopover1 } from "../popOverDemo1/page";
import { CardWithForm } from "../card/page";
import { AccordionDemo } from "../accordion";
import { ThemeProvider } from "../../components/ui/ThemeProvider";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import { ComboboxPopover2 } from '../popOverDemo2/page';
import { Skeleton } from "@/components/ui/skeleton"
import { ComboboxPopover3 } from '../popOverDemo3/page';
import Home from '../page';


const NavigationDemo = () => {
  const pathname = usePathname();
  const isMobile = useMediaQuery("(max-width: 768px)");

  const isResizingRef = useRef(false);
  const sidebarRef = useRef<ElementRef<"aside">>(null);
  const navbarRef = useRef<ElementRef<"div">>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const [isResetting, setIsResetting] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(isMobile);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageData, setImageData] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    if (file && file.type === "text/csv") {
      setCsvFile(file);
    } else {
      console.error("Please upload a valid CSV file.");
      setCsvFile(null);
    }
  };

  const handleSubmit = () => {
    if (!csvFile) {
      console.error("Please upload a CSV file.");
      return;
    }

    // Process the CSV file as needed
    console.log("CSV file:", csvFile);
  };

  const handleDownloadPDF = async () => {
    if (cardRef.current) {
      const { width, height } = cardRef.current.getBoundingClientRect();
      const canvas = await html2canvas(cardRef.current, { width, height });
      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF({
        unit: "px",
        format: [width, height], // Set the format to match the card dimensions
      });
      pdf.addImage(imgData, "PNG", 0, 0, width, height, "", "FAST");
      pdf.save("download.pdf");
    }
  };

  const handleFetchSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault(); // Prevent default form submission behavior
    setError(null); // Reset error state

    try {
      const response = await fetch('http://localhost:5000/kriging', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setResult(data.result);
    } catch (error) {
      console.error('Error:', error);
      setError('An error occurred while processing your request.');
    }
  };

  useEffect(() => {
    if (isMobile) {
      collapse();
    } else {
      resetWidth();
    }
  }, [isMobile]);

  useEffect(() => {
    if (isMobile) {
      collapse();
    }
  }, [pathname, isMobile]);

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    event.preventDefault();
    event.stopPropagation();

    isResizingRef.current = true;
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleMouseMove = (event: MouseEvent) => {
    if (!isResizingRef.current) return;
    let newWidth = event.clientX;

    if (newWidth < 240) newWidth = 240;
    if (newWidth > 480) newWidth = 480;

    if (sidebarRef.current && navbarRef.current) {
      sidebarRef.current.style.width = `${newWidth}px`;
      navbarRef.current.style.setProperty("left", `${newWidth}px`);
      navbarRef.current.style.setProperty("width", `calc(100% - ${newWidth}px)`);
    }
  };

  const handleMouseUp = () => {
    isResizingRef.current = false;
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);
  };

  const resetWidth = () => {
    if (sidebarRef.current && navbarRef.current) {
      setIsCollapsed(false);
      setIsResetting(true);

      sidebarRef.current.style.width = isMobile ? "100%" : "240px";
      navbarRef.current.style.setProperty("width", isMobile ? "0" : "calc(100% - 240px)");
      navbarRef.current.style.setProperty("left", isMobile ? "100%" : "240px");
      setTimeout(() => setIsResetting(false), 300);
    }
  };

  const collapse = () => {
    if (sidebarRef.current && navbarRef.current) {
      setIsCollapsed(true);
      setIsResetting(true);
      sidebarRef.current.style.width = "0";
      navbarRef.current.style.setProperty("width", "100%");
      navbarRef.current.style.setProperty("left", "0");
      setTimeout(() => setIsResetting(false), 300);
    }
  };

  type Status = {
    value: string
    label: string
  }

  const data1: Status[] = [
    { value: "Spatial", label: "Spatial Data" },
    { value: "Temporal", label: "Temporal Data" }
  ];

  const data2: Status[] = [
    { value: "Alpha", label: "Alpha Data" },
    { value: "Beta", label: "Beta Data" }
  ];



  return (
    <>
      <ThemeProvider
        attribute="class"
        defaultTheme="system"
        enableSystem
        disableTransitionOnChange
      >
        <aside
          ref={sidebarRef}
          className={cn(
            "group/sidebar h-screen bg-secondary overflow-y-auto relative flex w-60 flex-col z-[99999]",
            isResetting && "transition-all ease-in-out duration-300",
            isMobile && "w-0"
          )}
        >
          <div
            onClick={collapse}
            role="button"
            className={cn(
              "h-6 w-6 text-muted-foreground rounded-sm hover:bg-neutral-300 dark:hover:bg-neutral-600 absolute top-3 right-2 opacity-0 group-hover/sidebar:opacity-100 transition",
              isMobile && "opacity-100"
            )}
          >
            <ChevronLeftIcon className="h-6 w-6" />
          </div>

          <div
            onMouseDown={handleMouseDown}
            onClick={resetWidth}
            className="opacity-0 group-hover/sidebar:opacity-100 transition cursor-ew-resize absolute h-full w-1 bg-primary/10 right-0 top-0"
          >
            <p> </p>
          </div>
        </aside>
        <div
          ref={navbarRef}
          className={cn(
            "absolute top-0 z-[99999] left-60 w-[calc(100%-24px)]",
            isResetting && "transition-all ease-in-out duration-300",
            isMobile && "left-0 w-full"
          )}
        >
          <nav className="bg-transparent px-3 py-2 w-full">
            {isCollapsed && (
              <MenuIcon
                onClick={resetWidth}
                role="button"
                className="h-6 w-6 text-muted-foreground"
              />
            )}
          </nav>

          <NavigationMenuDemo />
          <div className="p-4 ml-20 flex-auto ">

            <Home/ >
            {/* <ComboboxPopover2 key="combo1" data={data1} />
            <ComboboxPopover2 key="combo2" data={data2} />
            <ComboboxPopover2 key="combo1" data={data1} />
            <ComboboxPopover2 key="combo2" data={data2} />
            <ComboboxPopover2 key="combo1" data={data1} />
            <ComboboxPopover2 key="combo2" data={data2} /> */}

            
            {/* <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              style={{ margin: "10px 0" }}
              className="mb-4"
            /><br />
            <ComboboxPopover1 /> <br />
            <ComboboxPopover /> <br />
            <form onClick={handleFetchSubmit}>
              <Button type="submit">Submit</Button>
            </form>
            <br />
            <div ref={cardRef}>
              <CardWithForm /><br />
            </div>
            <Button onClick={handleDownloadPDF}>Download as PDF</Button> <br />
            <Button>Zoom in</Button>
            <Button>Zoom out</Button> 
             */}
          </div>
        </div>
      </ThemeProvider>
    </>
  );
};

export default NavigationDemo; 