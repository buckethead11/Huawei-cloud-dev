import Navbar from "./Navbar";
import Features from "./Pages/Features";
import Customization from "./Pages/Customization"
import Home from "./Pages/Home"

function App() {
  let Component
  switch (window.location.pathname) {
    case "/":
      Component = Home
      break
    case "/features":
      Component = Features
      break
    case "/customization":
      Component = Customization
      break
  }
  return (
    <>
    <Navbar />
    <Component />
    </>
  );
}

export default App;
