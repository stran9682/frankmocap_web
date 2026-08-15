import { Canvas } from '@react-three/fiber'
import { useGLTF, OrbitControls, useAnimations } from '@react-three/drei'
import { Suspense, useEffect, useState, type ChangeEvent } from 'react'
import './global.css'
import './app.css'

const API_URL = import.meta.env.VITE_API_URL;

function Model ({url}:{url: string}){
  const gltf = useGLTF(url)
  
  const modelAnimations = useAnimations(gltf.animations, gltf.scene)
  
  useEffect(() => {
    modelAnimations.actions[modelAnimations.names[0]]?.play()
  }, [])

  return (<primitive object={gltf.scene} scale={[-2, 2, 2]}/>)
}

function App() {

  const [file, setFile] = useState<File|null>(null);
  const [modelUrl, setModelURL] = useState<string|null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  }

  const handleUpload = async () => {
    if (modelUrl) {
      URL.revokeObjectURL(modelUrl)
      setModelURL(null)
    }

    if (!file) {
      return
    }
    const extension = file.name.split('.').pop()!.toLowerCase();

    if (extension == 'gltf' || extension == 'glb') {
      const url = URL.createObjectURL(file)
      setModelURL(url)
    }
    else if (extension == 'mp4') {
      setIsLoading(true)
      const formData = new FormData()
      formData.append('uploaded_file', file)

      try {
        const response = await fetch(API_URL, {
          method: 'POST',
          body: formData
        });

        const blob = await response.blob();

        setModelURL(URL.createObjectURL(blob))

      } catch (error) {
        console.log(error)
      }
      finally {
        setIsLoading(false)
      }
    }
  }

  return (
    <div className='app space-mono-regular'>
      <input type="file" accept=".gltf, .glb, .mp4" onChange={handleFileChange}/>

      <button disabled={file === null || isLoading} onClick={handleUpload}>
        Upload File
      </button>

      {modelUrl ? 
          <Suspense>
            <Canvas>
              <ambientLight/>
              <pointLight position={[0, 2, 2]} intensity={10} />
              <Model url={modelUrl}/>
              <OrbitControls enablePan={true}/>
            </Canvas>
          </Suspense>
        :
        <div>
          <h2>{isLoading ? "Loading" : "Upload .mp4 or .glb"}</h2>
        </div>
      }
    </div>
  )
}

export default App
