export function getTheme() : string {
    return localStorage.getItem("theme") ?? "dark";
}

export function setTheme(theme:string) : void {
    if (theme !== "dark" && theme !== "light") { theme = "dark"; }
    localStorage.setItem("theme", theme);
}

export function flipTheme() : void {
    setTheme(getTheme() === "dark" ? "light" : "dark");
}

export function applyTheme(theme?:string) : void {
    if (!theme) { theme = getTheme(); }

    switch (theme) {
        case "dark":    document.documentElement.classList.add("dark"); break;
        case "light":   document.documentElement.classList.remove("dark"); break;
        default:        document.documentElement.classList.add("dark"); break;
    }
}