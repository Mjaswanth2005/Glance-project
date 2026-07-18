import { GooeyLoader } from "@/components/ui/loader-10"; // Adjust path as needed

export default function GooeyLoaderDemo() {
  return (
    // A minimal container to center the component for presentation.
    <div className="flex items-center justify-center w-full min-h-[250px]">
      <GooeyLoader
        primaryColor="#f87171" // red-400
        secondaryColor="#fca5a5" // red-300
        borderColor="#e5e7eb" // gray-200
      />
    </div>
  );
}
