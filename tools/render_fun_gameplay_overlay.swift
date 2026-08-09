import AppKit

let output = URL(fileURLWithPath: CommandLine.arguments[1])
let hook = CommandLine.arguments[2]
let payoff = CommandLine.arguments[3]
let image = NSImage(size: NSSize(width: 1080, height: 1920))
func color(_ red: CGFloat, _ green: CGFloat, _ blue: CGFloat, _ alpha: CGFloat = 1) -> NSColor { NSColor(red: red, green: green, blue: blue, alpha: alpha) }
func text(_ value: String, _ rect: NSRect, _ size: CGFloat, _ shade: NSColor) {
  let p = NSMutableParagraphStyle(); p.alignment = .center
  (value as NSString).draw(in: rect, withAttributes: [.font: NSFont.systemFont(ofSize: size, weight: .black), .foregroundColor: shade, .paragraphStyle: p])
}
image.lockFocus()
color(0.03, 0.03, 0.10, 0.82).setFill(); NSBezierPath(roundedRect: NSRect(x: 36, y: 1610, width: 1008, height: 240), xRadius: 28, yRadius: 28).fill()
text(hook, NSRect(x: 60, y: 1744, width: 960, height: 80), 58, .white)
text(payoff, NSRect(x: 60, y: 1664, width: 960, height: 62), 38, color(0.32, 0.90, 1.0))
color(0.03, 0.03, 0.10, 0.72).setFill(); NSBezierPath(roundedRect: NSRect(x: 36, y: 35, width: 1008, height: 72), xRadius: 18, yRadius: 18).fill()
text("XONOTIC 0.8.2 - DRUMMYFISH + DESARROLLADORES - GPLv3+", NSRect(x: 56, y: 58, width: 968, height: 28), 18, .white)
image.unlockFocus()
let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
try! bitmap.representation(using: .png, properties: [:])!.write(to: output)
