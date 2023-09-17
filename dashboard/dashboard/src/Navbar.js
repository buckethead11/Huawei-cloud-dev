export default function Navbar() {
    return (
        <nav className="nav">
            <a href='/' className="site-title">WeiFinder</a>
            <ul>
                <li>
                    <a href='/features'>Feautures</a>
                </li>
                <li>
                    <a href='/customization'>Customize</a>
                </li>
            </ul>
        </nav>
    )
}